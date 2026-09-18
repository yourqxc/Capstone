"""실험: 붙여넣은 합성본을 로컬 Stable Diffusion으로 다시 그리면 방에 녹아드는가 (DEVLOG §26).

기획서 7·8쪽의 "ControlNet + Stable Diffusion — 방 구조(Depth)를 유지하며 가구를 합성"을
기하 파이프라인의 **마지막 단계**로 붙여 본다. 기하 파이프라인이 만든 합성본을 초기 이미지로,
가구 주변을 다시 그릴 영역으로, 합성본의 깊이맵을 ControlNet 조건으로 넣는다.

    mode=whole  이미지 전체를 SD 해상도로 줄여 다시 그린다
    mode=crop   가구 주변(배치 박스의 2배, metrics.crop_around와 같은 규칙)만 잘라
                512px로 키워 다시 그리고 되돌려 붙인다

필요: pip install diffusers  (requirements.txt에는 넣지 않았다 — 앱은 쓰지 않는다)
모델: stable-diffusion-v1-5/stable-diffusion-inpainting + lllyasviel/control_v11f1p_sd15_depth
      fp16 약 2.9GB, 첫 실행 때 받는다. M5 16GB MPS에서 한 장 5~43초.

    python eval/experiments/sd_redraw.py --mode crop --strengths 0.35 0.6
    python eval/experiments/sd_redraw.py --mode whole --strengths 0.6 0.85
    python eval/experiments/sd_redraw.py --bed --mode whole --strengths 0.35 0.6 0.85
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.metrics import clip_delta  # noqa: E402
from eval.run_eval import BOXES, ITEMS, RESULTS, _next_run_name, pixel_box  # noqa: E402
from pipeline.compose import _warp_rgba, compose  # noqa: E402
from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import auto_height_px, floor_plane, place_transform  # noqa: E402
from pipeline.segment import cutout, pct_box  # noqa: E402

NEG = "floating, levitating, blurry, distorted, cartoon, painting, extra furniture, text, watermark"


def redraw_mask(alpha: np.ndarray) -> np.ndarray:
    """다시 그릴 영역: 가구 실루엣을 넓힌 것 + 가구 아래 바닥 띠(그림자·접지가 생길 자리)."""
    H, W = alpha.shape
    a = (alpha > 0.1).astype(np.uint8)
    m = cv2.dilate(a, np.ones((25, 25), np.uint8))
    ys, xs = np.where(a > 0)
    yb, yt = ys.max(), ys.min()
    cv2.rectangle(m, (max(0, xs.min() - 20), max(0, yb - int((yb - yt) * 0.25))),
                  (min(W - 1, xs.max() + 20), min(H - 1, yb + 30)), 1, -1)
    return m


def load_pipe():
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev != "cpu" else torch.float32
    cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", controlnet=cn, variant="fp16",
        torch_dtype=dtype, safety_checker=None, requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def redraw(pipe, comp: Image.Image, mask: np.ndarray, box, prompt: str, strength: float, mode: str) -> Image.Image:
    W, H = comp.size
    ctrl = Image.fromarray((np.clip(estimate_depth(comp), 0, 1) * 255).astype(np.uint8)).convert("RGB")
    mimg = Image.fromarray(mask * 255)
    if mode == "crop":
        x0, y0, x1, y1 = box
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
        L, T, R, B = max(0, int(cx - half)), max(0, int(cy - half)), min(W, int(cx + half)), min(H, int(cy + half))
    else:
        L, T, R, B = 0, 0, W, H
    cw, ch = R - L, B - T
    sc = 512 / max(cw, ch) if mode == "crop" else 1.0
    GW, GH = max(8, int(cw * sc) // 8 * 8), max(8, int(ch * sc) // 8 * 8)
    prep = lambda im: im.crop((L, T, R, B)).resize((GW, GH), Image.LANCZOS)
    out = pipe(prompt=prompt, negative_prompt=NEG, image=prep(comp), mask_image=prep(mimg),
               control_image=prep(ctrl), strength=strength, num_inference_steps=25, guidance_scale=7.5,
               controlnet_conditioning_scale=0.8,
               generator=torch.Generator("cpu").manual_seed(0)).images[0].resize((cw, ch), Image.LANCZOS)
    full = np.asarray(comp).astype(np.float32).copy()
    full[T:B, L:R] = np.asarray(out).astype(np.float32)
    fm = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)[:, :, None]   # 영역 밖은 원본 그대로
    res = np.asarray(comp).astype(np.float32) * (1 - fm) + full * fm
    return Image.fromarray(res.clip(0, 255).astype(np.uint8))


def jobs(args):
    if args.bed:     # 사용자 사진. 방 바닥에 침대 크기만큼, 침대 사진은 넉넉하게 칠했다고 가정
        room = Image.open(ROOT.parent / "Sample/방.jpg").convert("RGB")
        item = Image.open(ROOT.parent / "Sample/침대.jpg").convert("RGB")
        W, H = room.size
        box = (int(W * 0.25), int(H * 0.45), int(W * 0.80), int(H * 0.95))
        ib = (int(item.width * 0.12), int(item.height * 0.15), int(item.width * 0.88), int(item.height * 0.85))
        yield "bed", room, item, box, ib, None, "gray upholstered bed with pillows and a gray blanket"
        return
    for i in args.pairs:
        room = Image.open(ROOT / f"samples/rooms/room_{i:02d}.png").convert("RGB")
        item = Image.open(ROOT / f"samples/items/item_{i:02d}.png").convert("RGB")
        meta = ITEMS[i - 1]
        yield (f"room_{i:02d}__item_{i:02d}", room, item, pixel_box(room, BOXES[i - 1]),
               pct_box(item.size, meta["box"]), meta["height_m"], meta["en"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["whole", "crop"], default="crop")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.35, 0.6])
    ap.add_argument("--pairs", type=int, nargs="+", default=[2, 3, 6, 8],
                    help="샘플 번호. 기본은 누끼가 정상인 4쌍(결과를 보기 전에 고정)")
    ap.add_argument("--bed", action="store_true", help="Sample/방.jpg + 침대.jpg 로 한 장")
    args = ap.parse_args()

    out_root = RESULTS / _next_run_name(RESULTS, f"sd-{args.mode}")
    out_root.mkdir(parents=True)
    pipe = load_pipe()
    print(f"{'쌍':<20}{'강도':>6}{'초':>7}{'CLIP(영역) 합성→SD':>22}")
    for name, room, item, box, ib, h_m, en in jobs(args):
        depth = estimate_depth(room)
        plane = floor_plane(room, depth)
        rgba = cutout(item, box=ib, crop=True)
        hp = auto_height_px(plane, box[3], room.size, h_m) if h_m else None
        M = place_transform(plane, box, rgba.size, room.size, height_px=hp)
        comp = compose(room, rgba, M, plane, depth=depth)
        mask = redraw_mask(_warp_rgba(rgba, M, room.size)[1])
        comp.save(out_root / f"{name}_0.png")
        base = clip_delta(room, comp, en, box)["clip_delta"]
        prompt = f"a {en} standing on the floor of a room, photorealistic, natural lighting, soft contact shadow"
        for st in args.strengths:
            t = time.time()
            res = redraw(pipe, comp, mask, box, prompt, st, args.mode)
            dt = time.time() - t
            res.save(out_root / f"{name}_{st}.png")
            c = clip_delta(room, res, en, box)["clip_delta"]
            print(f"{name:<20}{st:>6}{dt:>7.1f}{base:>+12.4f} → {c:+.4f}", flush=True)
    print(f"\n저장: {out_root}  (이미지는 gitignore)")


if __name__ == "__main__":
    main()
