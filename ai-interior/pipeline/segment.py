"""가구 사진에서 객체만 분리한다 (SAM).

기획서 7쪽 "객체 분리 / 배경 제거 + 객체 마스킹" 대응.

프롬프트 방식은 실측으로 정했다. 샘플 가구 10점 기준:
    사용자 박스 7/10  >  중앙점+최대면적 5/10  >  이미지 전체 박스 2/10
중앙점은 부품만 따고(등받이만), 전체 박스는 배경을 객체로 잡는다.

SAM은 후보 마스크 3개를 내놓는데, 그중 "객체"와 "박스 안의 배경"을 자동으로
가르는 방법은 찾지 못했다. 샘플 10점에서 세 신호를 모두 측정한 결과:
    면적          - 배경이 더 큰 경우가 있다 (item_10: 배경 36.5% vs 객체 11.6%)
    이미지 테두리 - 박스를 안쪽으로 잡으면 모든 후보가 0.000으로 동일
    박스 변 피복률 - 배경(item_08 0.102)이 객체(item_07 0.264)보다 낮다
따라서 기본값은 SAM 자신의 IoU 점수를 쓰고, 후보 선택은 사용자에게 맡긴다
(candidate 인자). SAM 공식 데모도 같은 방식이다.

단독 실행:
    python pipeline/segment.py                 # samples/items 전체
    python pipeline/segment.py samples/items/item_05.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_ID = "facebook/sam-vit-base"
_MODEL = None
_PROC = None


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load():
    """모델은 처음 쓸 때 한 번만 올린다 (로드 약 14초)."""
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import SamModel, SamProcessor

        _PROC = SamProcessor.from_pretrained(MODEL_ID)
        _MODEL = SamModel.from_pretrained(MODEL_ID).to(device())
    return _MODEL, _PROC


def default_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """사용자가 박스를 안 그렸을 때 쓰는 중앙 80% 영역."""
    w, h = size
    return int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.90)


def _border_ratio(mask: np.ndarray) -> float:
    """마스크가 이미지 테두리를 얼마나 덮는가. 명백한 배경만 걸러내는 약한 신호."""
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(edges.mean())


def _clean(mask: np.ndarray) -> np.ndarray:
    """가장 큰 덩어리만 남기고 구멍을 메운다."""
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # 테두리에서 flood fill 해서 바깥이 아닌 구멍만 채운다
    ff = m.copy()
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    return (m | cv2.bitwise_not(ff)) > 0


def segment(item: Image.Image, box: tuple[int, int, int, int] | None = None,
            candidate: int | None = None) -> np.ndarray:
    """가구 마스크(bool). candidate로 SAM 후보 0/1/2를 직접 고를 수 있다."""
    model, proc = _load()
    if box is None:
        box = default_box(item.size)

    inputs = proc(item.convert("RGB"), input_boxes=[[list(box)]], return_tensors="pt")
    # MPS는 float64를 지원하지 않는다
    inputs = {k: (v.to(torch.float32) if v.dtype == torch.float64 else v) for k, v in inputs.items()}
    sizes = (inputs["original_sizes"], inputs["reshaped_input_sizes"])

    with torch.no_grad():
        out = model(**{k: v.to(device()) for k, v in inputs.items()})

    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), *sizes)[0][0]
    masks = masks.numpy().astype(bool)
    scores = out.iou_scores.cpu()[0][0].numpy()

    if candidate is not None:
        return _clean(masks[int(candidate) % len(masks)])

    # 이미지 테두리를 절반 넘게 덮으면 배경을 잡은 것이 확실하므로 제외한다.
    # (박스를 이미지 안쪽으로 잡으면 이 조건은 거의 걸리지 않는다.)
    ok = [i for i in range(len(masks)) if _border_ratio(masks[i]) < 0.5]
    if not ok:
        ok = [int(scores.argmax())]
        masks[ok[0]] = ~masks[ok[0]]
    return _clean(masks[max(ok, key=lambda i: scores[i])])


def cutout(item: Image.Image, box: tuple[int, int, int, int] | None = None,
           candidate: int | None = None, crop: bool = True, feather: int = 2) -> Image.Image:
    """가구만 남긴 RGBA. crop=True면 객체 경계로 잘라 반환한다 (합성 단계에서 쓰기 편하다)."""
    mask = segment(item, box, candidate)
    alpha = (mask.astype(np.uint8)) * 255
    if feather:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)

    rgba = item.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    if not crop:
        return rgba

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rgba
    return rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or sorted(Path("samples/items").glob("*.png"))
    out_dir = Path("samples/items_cutout")
    out_dir.mkdir(exist_ok=True)
    print(f"device={device()}  대상 {len(targets)}장")
    for p in targets:
        img = Image.open(p).convert("RGB")
        rgba = cutout(img, crop=True)
        rgba.save(out_dir / f"{p.stem}_cutout.png")
        area = np.array(rgba.split()[-1]).mean() / 255
        print(f"  {p.name:<14} -> {rgba.size}  알파 점유 {area*100:.1f}%")
        for c in range(3):      # 후보 3개도 남긴다 (사용자 선택 UI용 확인)
            cutout(img, candidate=c, crop=True).save(out_dir / f"{p.stem}_cand{c}.png")
    print(f"저장 위치: {out_dir}")
