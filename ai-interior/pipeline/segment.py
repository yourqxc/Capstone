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


MIN_MARGIN = 0.08   # 프롬프트 박스가 이미지 가장자리에서 떨어져 있어야 하는 최소 비율


def default_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """사용자가 박스를 안 그렸을 때 쓰는 중앙 80% 영역."""
    w, h = size
    return int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.90)


def clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """박스를 이미지 가장자리에서 최소 MIN_MARGIN 만큼 떼어 놓는다.

    박스가 이미지 전체에 가까워지면 SAM이 가구 대신 배경을 객체로 잡는다.
    침대 사진으로 여백을 훑어본 결과 경계가 뚜렷했다 (귀퉁이 피복률):
        여백 0%  0.995   3%  1.000   5%  0.631   <- 배경
        여백 8%  0.086  10%  0.000  16%  0.000   <- 정상
    """
    w, h = size
    mx, my = int(w * MIN_MARGIN), int(h * MIN_MARGIN)
    x0, y0, x1, y1 = box
    x0, y0 = max(x0, mx), max(y0, my)
    x1, y1 = min(x1, w - mx), min(y1, h - my)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return default_box(size)
    return x0, y0, x1, y1


def background_risk(mask: np.ndarray, box: tuple[int, int, int, int], frac: float = 0.08) -> float:
    """마스크가 프롬프트 박스의 네 귀퉁이를 덮는 정도. 높으면 배경을 잡은 것이다.

    샘플 가구 10점에서는 최대 0.508이었고 배경을 잡은 사례는 0.63~1.00이었다.
    면적·이미지 테두리·박스 채움률은 모두 판별에 실패했지만(§6) 이 신호는 갈렸다.
    """
    x0, y0, x1, y1 = box
    cw, ch = max(int((x1 - x0) * frac), 3), max(int((y1 - y0) * frac), 3)
    return float(np.mean([mask[y0:y0 + ch, x0:x0 + cw].mean(), mask[y0:y0 + ch, x1 - cw:x1].mean(),
                          mask[y1 - ch:y1, x0:x0 + cw].mean(), mask[y1 - ch:y1, x1 - cw:x1].mean()]))


def _border_ratio(mask: np.ndarray) -> float:
    """마스크가 이미지 테두리를 얼마나 덮는가. 명백한 배경만 걸러내는 약한 신호."""
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(edges.mean())


def _clean(mask: np.ndarray, max_hole_ratio: float = 0.005) -> np.ndarray:
    """가장 큰 덩어리만 남기고 작은 구멍만 메운다."""
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # 구멍은 "작은 것만" 메운다.
    # 의자 등받이의 틈, 테이블 다리 사이처럼 실제로 뚫린 공간까지 메우면
    # 배경이 가구 안으로 딸려 들어온다 (item_02에서 불투명 픽셀의 13.2%가 초록 배경이었다).
    holes = (m == 0).astype(np.uint8)
    nh, hl, hs, _ = cv2.connectedComponentsWithStats(holes, 4)
    outside = set(np.unique(hl[[0, -1], :])) | set(np.unique(hl[:, [0, -1]]))
    limit = max_hole_ratio * m.size
    small = [i for i in range(1, nh)
             if i not in outside and hs[i, cv2.CC_STAT_AREA] < limit]
    if small:
        m = np.where(np.isin(hl, small), 255, m).astype(np.uint8)
    return m > 0


def segment(item: Image.Image, box: tuple[int, int, int, int] | None = None,
            candidate: int | None = None) -> np.ndarray:
    """가구 마스크(bool). candidate로 SAM 후보 0/1/2를 직접 고를 수 있다."""
    model, proc = _load()
    box = default_box(item.size) if box is None else clamp_box(box, item.size)

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


def has_alpha(item: Image.Image, min_transparent: float = 0.05) -> bool:
    """이미 배경이 제거된 이미지인지. 투명 픽셀이 충분히 있으면 참."""
    if item.mode not in ("RGBA", "LA"):
        return False
    a = np.asarray(item.convert("RGBA"))[:, :, 3]
    return bool((a < 16).mean() >= min_transparent)


def cutout(item: Image.Image, box: tuple[int, int, int, int] | None = None,
           candidate: int | None = None, crop: bool = True, feather: int = 2) -> Image.Image:
    """가구만 남긴 RGBA. crop=True면 객체 경계로 잘라 반환한다 (합성 단계에서 쓰기 편하다).

    이미 알파 채널이 있는 이미지(누끼된 PNG)는 SAM을 돌리지 않고 그대로 쓴다.
    쇼핑몰 제품컷은 대부분 투명 PNG이고, 그 알파가 SAM 추정보다 정확하다.
    """
    if has_alpha(item):
        mask = np.asarray(item.convert("RGBA"))[:, :, 3] > 127
    else:
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
