"""선택 단계: 기하 합성본을 로컬 Stable Diffusion으로 다듬는다 (DEVLOG §26, §27).

기하 파이프라인이 무엇을 어디에 어떤 크기로 놓을지 정하고, 이 단계는 가구 주변의
경계·색·그림자만 다시 그린다. 기획서 8쪽 "ControlNet — 방 구조(Depth)를 유지하며 합성".

가구 주변(배치 박스의 2배)만 잘라 512px로 키워 인페인팅하고 되돌려 붙인다. 이미지를
통째로 넣으면 100~150px짜리 가구에 픽셀이 모자라 다른 물건이 됐다(보존 0/4 → 4/4).
강도가 높을수록 자연스러워지고 가구가 바뀐다. 기본값 0.35는 샘플 4쌍 모두 가구가
유지된 값이다.

필요: pip install diffusers. 모델(fp16 약 2.9GB)은 처음 켤 때 받는다.
M5 16GB MPS에서 한 장 5~13초, 메모리 약 7GB.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
from PIL import Image

from pipeline.depth import estimate_depth

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
CN_MODEL = "lllyasviel/control_v11f1p_sd15_depth"
DEFAULT_STRENGTH = 0.35
NEG = "floating, levitating, blurry, distorted, cartoon, painting, extra furniture, text, watermark"

_PIPE = None


def available() -> bool:
    try:
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _PIPE
    if _PIPE is None:
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev != "cpu" else torch.float32
        cn = ControlNetModel.from_pretrained(CN_MODEL, variant="fp16", torch_dtype=dtype)
        _PIPE = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_MODEL, controlnet=cn, variant="fp16", torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False).to(dev)
        _PIPE.set_progress_bar_config(disable=True)
    return _PIPE


def redraw_mask(alpha: np.ndarray) -> np.ndarray:
    """다시 그릴 영역(0/1): 가구 실루엣을 넓힌 것 + 가구 아래 바닥 띠(그림자·접지가 생길 자리)."""
    H, W = alpha.shape
    a = (alpha > 0.1).astype(np.uint8)
    m = cv2.dilate(a, np.ones((25, 25), np.uint8))
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return m
    yb, yt = ys.max(), ys.min()
    cv2.rectangle(m, (max(0, xs.min() - 20), max(0, yb - int((yb - yt) * 0.25))),
                  (min(W - 1, xs.max() + 20), min(H - 1, yb + 30)), 1, -1)
    return m


def prompt_for(item_en: str | None) -> str:
    what = item_en or "piece of furniture"
    return f"a {what} standing on the floor of a room, photorealistic, natural lighting, soft contact shadow"


def refine(comp: Image.Image, alpha: np.ndarray, box, item_en: str | None = None,
           strength: float = DEFAULT_STRENGTH, mode: str = "crop", seed: int = 0) -> Image.Image:
    """합성본의 가구 주변만 다시 그린다. 영역 밖 픽셀은 원본 그대로다.

    mode="crop"   배치 박스 2배 영역을 512px로 키워 그린다 (앱 기본)
    mode="whole"  이미지 전체를 넣는다 (실험 비교용 — 작은 가구가 바뀐다)
    """
    W, H = comp.size
    mask = redraw_mask(alpha)
    ctrl = Image.fromarray((np.clip(estimate_depth(comp), 0, 1) * 255).astype(np.uint8)).convert("RGB")
    if mode == "crop":
        x0, y0, x1, y1 = box
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
        L, T, R, B = max(0, int(cx - half)), max(0, int(cy - half)), min(W, int(cx + half)), min(H, int(cy + half))
        sc = 512 / max(R - L, B - T)
    else:
        L, T, R, B, sc = 0, 0, W, H, 1.0
    cw, ch = R - L, B - T
    GW, GH = max(8, int(cw * sc) // 8 * 8), max(8, int(ch * sc) // 8 * 8)
    prep = lambda im: im.crop((L, T, R, B)).resize((GW, GH), Image.LANCZOS)
    out = _load()(prompt=prompt_for(item_en), negative_prompt=NEG, image=prep(comp),
                  mask_image=prep(Image.fromarray(mask * 255)), control_image=prep(ctrl),
                  strength=strength, num_inference_steps=25, guidance_scale=7.5,
                  controlnet_conditioning_scale=0.8,
                  generator=torch.Generator("cpu").manual_seed(seed)).images[0].resize((cw, ch), Image.LANCZOS)
    full = np.asarray(comp).astype(np.float32).copy()
    full[T:B, L:R] = np.asarray(out).astype(np.float32)
    fm = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)[:, :, None]
    res = np.asarray(comp).astype(np.float32) * (1 - fm) + full * fm
    return Image.fromarray(res.clip(0, 255).astype(np.uint8))
