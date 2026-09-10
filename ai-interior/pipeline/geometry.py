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
                iters: int = 400, tol: float = 0.022, seed: int = 0) -> dict:
    """바닥 평면 추정. 소실선(지평선), 평면 계수, 바닥 마스크를 돌려준다.

    tol은 평면 허용오차다. 10개 방 정답 대비 IoU로 실측해 정했다 (float32 깊이 기준).
        0.014  0.764  (room_10 붕괴 - 바닥이 조각나 하나도 못 잡는다)
        0.018  0.863
        0.022  0.864  <- 안정 구간의 가운데
        0.026  0.861
        0.030  0.795  (room_04 붕괴 - 낮은 침대를 바닥으로 흡수한다)
    느슨하면 침대 윗면과 벽을 바닥에 흡수하고, 너무 좁으면 바닥을 놓친다.

    **깊이맵 정밀도는 병목이 아니었다.** 8비트 시각화 출력(고유값 231개)에서
    float32 원본(30만 개 이상)으로 바꾸고 다시 훑었으나 최적값과 평균 IoU가
    0.022 / 0.864 로 동일했다. 양자화 계단 폭 안에서 튜닝하고 있다는 우려가
    있었지만 실제 제약은 깊이 정밀도가 아니라 깊이 모델의 정확도와 바닥 가림이다.
    """
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


def _disp(plane: dict, x: float, y: float, size: tuple[int, int]) -> float:
    a, b, c = plane["coef"]
    W, H = size
    return a * (x / W) + b * (y / H) + c


def depth_at(plane: dict, x: float, y: float, size: tuple[int, int]) -> float:
    """평면 위 한 점의 상대 깊이(값이 클수록 가깝다)."""
    return _disp(plane, x, y, size)


def place_transform(plane: dict, box: tuple[int, int, int, int],
                    item_size: tuple[int, int], room_size: tuple[int, int],
                    mode: str = "upright", scale: float = 1.0,
                    height_px: float | None = None) -> np.ndarray:
    """배치 박스를 바닥 평면 위 원근에 맞춘 3x3 호모그래피 (가구 이미지 -> 방 이미지).

    물리적으로 두 경우가 다르다.

    upright (의자·책장처럼 세워지는 가구)
        바닥에 서 있는 수직 빌보드다. 원근 왜곡은 일어나지 않고 **크기만** 변한다.
        고정 실제 높이 h의 물체가 이미지에서 갖는 높이는 역깊이에 비례하므로
        (h_px = f·h/Z = f·h·disp), 접지점의 disp가 크기를 결정한다.
        따라서 사다리꼴이 아니라 사각형이며, 가구의 종횡비를 지킨다.

    flat (러그처럼 바닥에 깔리는 것)
        바닥 평면 위의 사각형이므로 진짜 원근 사다리꼴이 된다.
        먼 쪽(위) 변이 disp 비율만큼 좁아진다.
    """
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    iw, ih = item_size
    cx = (x0 + x1) / 2

    if mode == "flat":
        d_near = max(_disp(plane, cx, y1, room_size), 1e-4)
        d_far = max(_disp(plane, cx, y0, room_size), 1e-4)
        shrink = float(np.clip(d_far / d_near, 0.15, 1.0))   # 먼 쪽이 좁아진다
        half = bw / 2
        dst = np.float32([
            [cx - half * shrink, y0], [cx + half * shrink, y0],
            [x1, y1], [x0, y1],
        ])
    else:
        # 크기의 출처는 둘 중 하나다.
        #   height_px 가 주어지면 = 바닥 평면과 지평선에서 계산한 물리적 크기 (자동)
        #   없으면 = 사용자가 칠한 박스 높이 (수동)
        h = float(height_px) if height_px else bh
        w = h * iw / ih
        if w > bw * 1.6:                 # 박스보다 지나치게 넓어지면 폭에 맞춘다
            w = bw
            h = w * ih / iw
        h *= scale                        # 사용자 크기 배율 (접지점은 그대로 둔다)
        w *= scale
        dst = np.float32([
            [cx - w / 2, y1 - h], [cx + w / 2, y1 - h],
            [cx + w / 2, y1], [cx - w / 2, y1],
        ])

    src = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]])
    return cv2.getPerspectiveTransform(src, dst)


def scale_hint(plane: dict, y: float, room_size: tuple[int, int],
               ref_y: float | None = None) -> float:
    """같은 가구를 y 위치에 놓을 때의 상대 크기. 원근 일관성 확인용."""
    W, H = room_size
    ref_y = H * 0.95 if ref_y is None else ref_y
    d_ref = max(_disp(plane, W / 2, ref_y, room_size), 1e-4)
    return float(max(_disp(plane, W / 2, y, room_size), 1e-4) / d_ref)


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


CAM_HEIGHT_M = 1.4      # 실내 사진의 통상 촬영 눈높이. 방마다 다르지만 이 가정으로 9/10이 맞았다.


def auto_height_px(plane: dict, y_contact: float, room_size: tuple[int, int],
                   height_m: float, cam_height_m: float = CAM_HEIGHT_M) -> float | None:
    """바닥에 선 가구의 화면 높이를 **지평선에서** 계산한다. 초점거리를 몰라도 된다.

    핀홀 카메라에서 바닥에 선 높이 h의 물체는

        h_px / (y_접지 - y_지평선) = h / H_카메라

    를 만족한다. 좌변의 분모는 접지점이 지평선에서 얼마나 떨어졌는지이고,
    그것이 곧 거리 정보다. 따라서 지평선만 있으면 크기가 정해진다.

    이것이 기획서 6쪽 "원근·크기 자동 보정"의 실체다. 이전에는 place_transform이
    평면을 전혀 참조하지 않고 사용자 박스 높이만 썼기 때문에, 같은 가구를 방 앞뒤로
    옮겨도 화면 크기가 같았다.

    검증(0.9m 의자를 바닥 90% 지점에): 방 10개 중 9개에서 화면의 16.6~29.7%로
    그럴듯한 값이 나왔다. 실패한 room_09는 바닥 IoU가 0.679로 가장 낮아
    지평선 추정 자체가 부정확한 방이다.

    지평선이 없거나 접지점이 지평선 위면 None을 돌려준다 — 호출부가 박스 높이로 되돌아간다.
    """
    hy = plane.get("horizon_y") if plane else None
    if hy is None:
        return None
    gap = float(y_contact) - float(hy)
    if gap <= 1.0:                      # 접지가 지평선 위 = 바닥에 놓일 수 없는 위치
        return None
    return gap * float(height_m) / float(cam_height_m)
