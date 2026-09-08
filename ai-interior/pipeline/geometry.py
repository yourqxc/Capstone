"""방 사진에서 바닥 평면과 원근을 추정한다 (OpenCV + 깊이맵).

기획서 7쪽 "원근 변환 행렬 계산", 8쪽 "OpenCV — 이미지 전처리, 원근 변환 및
기하학적 데이터 처리" 대응.

원리. 핀홀 카메라로 평면을 보면 **역깊이(disparity)가 이미지 좌표의 1차식**이 된다.

    disp(x, y) = a·x + b·y + c

Depth Anything V2의 출력은 역깊이에 비례하므로, 깊이맵에서 이 1차식을 만족하는
픽셀 집합을 찾으면 그게 바닥이다. RANSAC으로 찾는다.
바닥은 아래로 갈수록 가까우므로 b > 0 이어야 하고, 이 조건이 벽·천장을 걸러낸다.

지평선은 그 평면의 깊이가 0이 되는 선이다: a·x + b·y + c = 0.

단독 실행:
    python pipeline/geometry.py            # samples/rooms 전체, 정답 대비 IoU 출력
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _fit_plane(xs, ys, ds):
    """최소제곱으로 disp = a·x + b·y + c 를 푼다."""
    A = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    coef, *_ = np.linalg.lstsq(A, ds, rcond=None)
    return coef  # (a, b, c)


def _floor_components(mask: np.ndarray) -> np.ndarray:
    """하단에 닿은 덩어리를 모두 남긴다.

    가구에 가려 바닥이 여러 조각으로 끊기므로 가장 큰 덩어리 하나만 남기면 안 된다
    (침대 두 개 사이의 바닥이 통째로 날아간다).

    "닿지 않아도 충분히 큰 덩어리는 남긴다"는 규칙도 시험했으나 벽까지 들어와
    10개 방 중 4개에서 오히려 손해였다 (평균 IoU 하단접촉 0.679 > 하단+대형 0.656).
    """
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return m > 0
    keep = set(np.unique(labels[-3:, :])) - {0}
    if not keep:
        keep = {max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])}
    return np.isin(labels, list(keep))


def floor_plane(room: Image.Image, depth: np.ndarray | None = None,
                iters: int = 400, tol: float = 0.035, seed: int = 0) -> dict:
    """바닥 평면 추정. 소실선(지평선), 평면 계수, 바닥 마스크를 돌려준다."""
    if depth is None:
        from pipeline.depth import estimate_depth
        depth = estimate_depth(room)

    H, W = depth.shape
    # 속도를 위해 축소해서 맞추고, 마스크만 원래 크기로 되돌린다
    sw = 200
    sh = max(1, int(H * sw / W))
    d = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xn, yn = xx / sw, yy / sh                      # 좌표 정규화 (계수 스케일 안정)

    # 바닥일 가능성이 높은 하단 영역에서 표본을 뽑는다
    seed_mask = yn > 0.55
    sy, sx, sd = yn[seed_mask], xn[seed_mask], d[seed_mask]
    rng = np.random.default_rng(seed)

    # 바닥을 다른 수평면(침대 윗면 등)과 구분하는 두 가지 제약:
    #  (1) 바닥은 화면 맨 아래 줄을 채운다 — 침대는 그렇지 않다
    #  (2) 그 평면의 지평선이 화면 안에 있어야 한다 — 벽/천장을 잡으면 화면 밖으로 나간다
    best = (0.0, None)
    flat_x, flat_y, flat_d = xn.ravel(), yn.ravel(), d.ravel()
    bottom = flat_y > 0.95
    for _ in range(iters):
        i = rng.integers(0, len(sd), 3)
        try:
            coef = _fit_plane(sx[i], sy[i], sd[i])
        except np.linalg.LinAlgError:
            continue
        a, b, c = coef
        if b <= 0:                # 아래로 갈수록 가까워야 바닥이다
            continue
        horizon = -(a * 0.5 + c) / b
        if not (0.0 <= horizon <= 0.95):        # 제약 (2)
            continue
        inl = np.abs(flat_d - (a * flat_x + b * flat_y + c)) < tol
        if inl[bottom].mean() < 0.5:            # 제약 (1)
            continue
        if inl.sum() > best[0]:
            best = (float(inl.sum()), coef)

    if best[1] is None:
        empty = np.zeros((H, W), bool)
        return {"mask": empty, "coef": None, "horizon_y": None, "inlier_ratio": 0.0}

    # 인라이어로 한 번 더 정밀하게 맞춘다
    coef = best[1]
    for _ in range(3):
        pred = coef[0] * flat_x + coef[1] * flat_y + coef[2]
        inl = np.abs(flat_d - pred) < tol
        if inl.sum() < 50:
            break
        refined = _fit_plane(flat_x[inl], flat_y[inl], flat_d[inl])
        a, b, c = refined
        if b <= 0 or not (0.0 <= -(a * 0.5 + c) / b <= 0.95):
            break                               # 정밀화가 제약을 깨면 직전 값을 쓴다
        coef = refined

    pred = (coef[0] * xn + coef[1] * yn + coef[2])
    small = (np.abs(d - pred) < tol) & (yn > 0.30)   # 지나치게 높은 곳은 바닥이 아니다
    small = _floor_components(small)
    mask = cv2.resize(small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    # 지평선: 평면의 깊이가 0이 되는 y (화면 중앙 x 기준)
    a, b, c = coef
    horizon = -(a * 0.5 + c) / b if abs(b) > 1e-9 else None
    return {
        "mask": mask,
        "coef": (float(a), float(b), float(c)),
        "horizon_y": float(horizon * H) if horizon is not None else None,
        "inlier_ratio": float(small.mean()),
    }


def depth_at(plane: dict, x: float, y: float, size: tuple[int, int]) -> float:
    """평면 위 한 점의 상대 깊이(값이 클수록 가깝다)."""
    a, b, c = plane["coef"]
    W, H = size
    return a * (x / W) + b * (y / H) + c


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = (pred & truth).sum()
    union = (pred | truth).sum()
    return float(inter / union) if union else 0.0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.depth import estimate_depth

    targets = [Path(a) for a in sys.argv[1:]] or sorted(Path("samples/rooms").glob("*.png"))
    out_dir = Path("samples/rooms_floor")
    out_dir.mkdir(exist_ok=True)

    scores = []
    print(f"{'파일':<14} {'IoU':>6} {'정밀도':>7} {'재현율':>7} {'지평선y':>8}  비고")
    for p in targets:
        room = Image.open(p).convert("RGB")
        depth = estimate_depth(room)
        pl = floor_plane(room, depth)
        pred = pl["mask"]

        gt_path = Path("samples/rooms_gt") / f"{p.stem}_gt.png"
        line = f"{p.name:<14}"
        if gt_path.exists():
            gt = np.array(Image.open(gt_path).resize(room.size, Image.NEAREST)) == 255
            v = iou(pred, gt)
            prec = (pred & gt).sum() / max(pred.sum(), 1)
            rec = (pred & gt).sum() / max(gt.sum(), 1)
            scores.append(v)
            line += f" {v:6.3f} {prec:7.3f} {rec:7.3f}"
        else:
            line += f" {'-':>6} {'-':>7} {'-':>7}"
        hy = pl["horizon_y"]
        line += f" {hy:8.0f}" if hy is not None else f" {'-':>8}"
        print(line)

        # 시각화: 추정 바닥을 초록으로 덮는다
        vis = np.array(room).astype(np.int16)
        vis[pred, 1] = np.minimum(255, vis[pred, 1] + 90)
        if hy is not None and 0 <= hy < room.size[1]:
            vis[int(hy):int(hy) + 2, :] = (255, 40, 40)
        Image.fromarray(vis.astype(np.uint8)).save(out_dir / f"{p.stem}_floor.png")

    if scores:
        print(f"\n평균 IoU {np.mean(scores):.3f}   중앙값 {np.median(scores):.3f}   "
              f"최저 {min(scores):.3f}   최고 {max(scores):.3f}")
    print(f"저장 위치: {out_dir}")
