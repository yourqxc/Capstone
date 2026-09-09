"""합성 결과를 정량 평가한다.

기획서 p7·p8의 "CLIP / Text Embedding" 대응.

두 지표를 쓰고, **역할이 다르다.**

1. CLIP score — 배치 영역 **안쪽 품질**. 실제로 우열이 갈리는 축이다.
   결과 이미지가 "그 가구가 놓인 방"이라는 문장에 얼마나 부합하는가.
   절대값보다 **원본 대비 증가분(delta)**이 의미 있다. 원본 방에도 다른 가구가
   있어서 절대 점수는 방마다 다르지만, delta는 "이 가구가 추가됐는가"를 잰다.

2. 마스크 밖 SSIM — **무결성 확인용. 우열 지표가 아니다.**
   로컬 파이프라인은 알파 합성이라 배치 영역 밖 픽셀이 원본과 비트 단위로 같다
   (실측 79.1%가 완전 동일). 생성 모델은 이미지 전체를 다시 그리므로 항상 1 미만이다.
   즉 이 지표는 로컬이 정의상 이긴다. "구조 보존이 우수하다"는 주장의 근거로 쓰면
   "알파 합성이니 당연"이라는 지적을 받는다.
   대신 **"의도한 영역만 바꿨는가"를 확인하는 체크**로만 쓴다.

단독 실행:
    python eval/metrics.py      # 자기 검증 — 지표가 실제로 구별하는지 확인한다
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

MODEL_ID = "openai/clip-vit-base-patch32"
_MODEL = None
_PROC = None


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load():
    """CLIP은 처음 쓸 때 한 번만 올린다 (로드 약 23초, 151M 파라미터)."""
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import CLIPModel, CLIPProcessor

        _MODEL = CLIPModel.from_pretrained(MODEL_ID).to(device()).eval()
        _PROC = CLIPProcessor.from_pretrained(MODEL_ID)
    return _MODEL, _PROC


def clip_score(image: Image.Image, text: str) -> float:
    """이미지와 영어 문장의 CLIP 코사인 유사도. 보통 0.15~0.35 범위에 들어온다."""
    model, proc = _load()
    inputs = proc(text=[text], images=image.convert("RGB"),
                  return_tensors="pt", padding=True, truncation=True).to(device())
    with torch.no_grad():
        out = model(**inputs)
        img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_e = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_e @ txt_e.T).squeeze().cpu())


def placement_text(item_en: str) -> str:
    """CLIP에 물어볼 문장. 모든 조건에 동일하게 쓴다."""
    return f"a photo of a room with a {item_en} placed on the floor"


def clip_delta(before: Image.Image, after: Image.Image, item_en: str) -> dict:
    """원본 대비 CLIP 점수 증가분. 양수면 그 가구가 실제로 추가된 것으로 본다."""
    text = placement_text(item_en)
    b, a = clip_score(before, text), clip_score(after, text)
    return {"clip_before": round(b, 4), "clip_after": round(a, 4), "clip_delta": round(a - b, 4)}


def ssim_outside(before: Image.Image, after: Image.Image, box) -> float:
    """배치 영역 **밖**의 구조 보존도 (0~1).

    무결성 확인용이다. 로컬 파이프라인은 1.0에 가깝게 나오는 것이 정상이고,
    그 사실 자체가 품질의 근거는 아니다. 모듈 상단 설명 참조.
    """
    a = np.asarray(before.convert("RGB"))
    b = np.asarray(after.convert("RGB").resize(before.size)) if after.size != before.size \
        else np.asarray(after.convert("RGB"))
    x0, y0, x1, y1 = (int(v) for v in box)
    mask = np.ones(a.shape[:2], bool)
    mask[max(y0, 0):y1, max(x0, 0):x1] = False        # 배치 영역 제외
    if mask.sum() < 100:
        return float("nan")
    _, smap = structural_similarity(a, b, channel_axis=2, full=True)
    return float(smap.mean(axis=2)[mask].mean())


def identical_ratio(before: Image.Image, after: Image.Image) -> float:
    """원본과 픽셀이 완전히 같은 비율. 합성 방식의 성격을 드러내는 보조 수치."""
    a = np.asarray(before.convert("RGB"))
    b = np.asarray(after.convert("RGB").resize(before.size)) if after.size != before.size \
        else np.asarray(after.convert("RGB"))
    return float((a == b).all(axis=2).mean())


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.compose import compose
    from pipeline.depth import estimate_depth
    from pipeline.geometry import floor_plane, place_transform
    from pipeline.segment import cutout

    room = Image.open("samples/rooms/room_03.png").convert("RGB")
    item = Image.open("samples/items/item_04.png").convert("RGB")
    W, H = room.size
    box = (int(W * 0.36), int(H * 0.60), int(W * 0.60), int(H * 0.92))

    plane = floor_plane(room, estimate_depth(room))
    rgba = cutout(item, crop=True)
    placed = compose(room, rgba, place_transform(plane, box, rgba.size, room.size), plane)

    en = "carved wooden armchair"
    print(f"검증 대상: room_03 + item_04 ({en})\n")
    print("① CLIP score — 가구를 넣으면 점수가 올라가야 한다")
    d = clip_delta(room, placed, en)
    print(f"   원본 {d['clip_before']:.4f} → 합성 {d['clip_after']:.4f}   delta {d['clip_delta']:+.4f}"
          f"   {'OK' if d['clip_delta'] > 0 else '실패 — 지표가 구별하지 못한다'}")

    print("\n② 엉뚱한 문장에는 반응하지 않아야 한다")
    for t in ["a photo of a room with a refrigerator placed on the floor",
              "a photo of an empty beach"]:
        b, a = clip_score(room, t), clip_score(placed, t)
        print(f"   {t[:52]:<52} delta {a - b:+.4f}")

    print("\n③ 마스크 밖 SSIM — 로컬은 1.0에 가까워야 정상 (무결성 확인)")
    print(f"   SSIM(배치영역 밖) {ssim_outside(room, placed, box):.4f}")
    print(f"   원본과 완전히 같은 픽셀 비율 {identical_ratio(room, placed) * 100:.1f}%")
