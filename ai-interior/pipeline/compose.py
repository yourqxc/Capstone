"""원근 보정된 가구를 방 사진에 합성한다 (OpenCV).

기획서 6쪽 "배치 합성 — 원근, 크기 자동 보정" 대응.

접지 그림자를 함께 만든다. 그림자가 없으면 가구가 바닥에 떠 보인다 —
합성 품질에서 가장 크게 차이 나는 요소다.
그림자는 가구 알파의 아래쪽을 바닥 평면 방향으로 눌러 만든 타원 근사이며,
물리 기반 렌더링이 아니라 접지감을 주기 위한 근사임을 밝혀둔다.

단독 실행:
    python pipeline/compose.py            # 방 x 가구 조합을 samples/compose/ 에 저장
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _warp_rgba(item_rgba: Image.Image, H: np.ndarray, size: tuple[int, int]):
    """가구 RGBA를 방 좌표계로 옮긴다. (rgb, alpha) 반환."""
    arr = np.array(item_rgba.convert("RGBA"))
    warped = cv2.warpPerspective(arr, H, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped[:, :, :3].astype(np.float32), warped[:, :, 3].astype(np.float32) / 255.0


def light_direction(room: Image.Image) -> float:
    """방의 밝기 분포로 광원 방향을 추정한다. -1(왼쪽에서) ~ +1(오른쪽에서).

    창이 있는 쪽이 밝다는 단순한 가정이다. 정확한 광원 추정이 아니라
    그림자를 어느 쪽으로 늘릴지 정하는 용도다.
    """
    g = np.asarray(room.convert("L"), dtype=np.float32)
    third = max(g.shape[1] // 3, 1)
    left, right = g[:, :third].mean(), g[:, -third:].mean()
    return float(np.clip((left - right) / max((left + right) / 2, 1.0) * 3.0, -1.0, 1.0))


def contact_shadow(alpha: np.ndarray, strength: float = 0.50, squash: float = 0.12,
                   spread: float = 1.06, drop: float = 0.06,
                   light: float = 0.0) -> np.ndarray:
    """접지 그림자.

    실루엣을 접지선 쪽으로 납작하게 누른 뒤 **아래로 조금 더 밀어낸다.**
    누르기만 하면 그림자가 가구 실루엣 뒤에 완전히 가려 보이지 않는다
    (그게 첫 구현의 문제였다 — DEVLOG §11).

    light: 광원 방향. 양수면 왼쪽에서 빛이 와 그림자가 오른쪽으로 늘어난다.
    """
    ys, xs = np.where(alpha > 0.3)
    if len(ys) == 0:
        return np.zeros_like(alpha)

    bottom, top = ys.max(), ys.min()
    height = max(bottom - top, 1)
    cx = (xs.min() + xs.max()) / 2
    shift_x = light * height * 0.10

    M = np.float32([
        [spread, 0, -(spread - 1) * cx + shift_x],
        [0, squash, bottom * (1 - squash) + drop * height],
    ])
    flat = cv2.warpAffine((alpha * 255).astype(np.uint8), M,
                          (alpha.shape[1], alpha.shape[0]))
    k = max(3, int(0.08 * height) * 2 + 1)
    flat = cv2.GaussianBlur(flat, (k, k), 0)
    return np.clip(flat.astype(np.float32) / 255.0 * strength, 0, 1)


def harmonize(rgb: np.ndarray, alpha: np.ndarray, room: Image.Image,
              region: np.ndarray | None = None, amount: float = 0.35) -> np.ndarray:
    """가구의 색조를 방의 조명 쪽으로 부분 이동시킨다 (LAB 평균 이동).

    컷아웃은 원래 촬영 조명을 그대로 들고 온다. 방이 따뜻한 자연광인데 가구가
    중성 스튜디오광이면 아무리 배치가 정확해도 붙여넣은 티가 난다.
    가구 고유의 색을 잃지 않도록 amount(기본 0.35)만큼만 옮긴다.
    """
    inside = alpha > 0.6
    if inside.sum() < 50:
        return rgb
    ref = np.asarray(room.convert("RGB"), dtype=np.uint8)
    ref_px = ref[region] if region is not None and region.sum() > 50 else ref.reshape(-1, 3)

    item_lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).astype(np.float32)

    shift = ref_lab.reshape(-1, 3).mean(axis=0) - item_lab[inside].mean(axis=0)
    shift[0] *= 0.5                       # 밝기는 절반만 — 형태감이 뭉개진다
    item_lab += shift * amount
    out = cv2.cvtColor(np.clip(item_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out.astype(np.float32)


def compose(room: Image.Image, item_rgba: Image.Image, H: np.ndarray,
            plane: dict | None = None, shadow: bool = True,
            harmonize_amount: float = 0.35) -> Image.Image:
    """가구를 방에 합성한다. H는 geometry.place_transform이 만든 호모그래피."""
    base = np.array(room.convert("RGB")).astype(np.float32)
    rgb, alpha = _warp_rgba(item_rgba, H, room.size)

    floor = plane.get("mask") if plane is not None else None
    if harmonize_amount > 0:
        rgb = harmonize(rgb, alpha, room, floor, harmonize_amount)

    if shadow:
        sh = contact_shadow(alpha, light=light_direction(room))
        if floor is not None:
            sh = sh * floor.astype(np.float32)           # 그림자는 바닥에만 진다
        base *= (1.0 - sh)[:, :, None]

    a = alpha[:, :, None]
    out = base * (1 - a) + rgb * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.depth import estimate_depth
    from pipeline.geometry import floor_plane, place_transform, scale_hint
    from pipeline.segment import cutout

    out_dir = Path("samples/compose")
    out_dir.mkdir(exist_ok=True)

    # 방 x 가구 조합. box는 (x0%, y0%, x1%, y1%), mode는 세움/눕힘
    JOBS = [
        ("room_03", "item_03", (0.34, 0.52, 0.56, 0.86), "upright"),
        ("room_06", "item_01", (0.13, 0.52, 0.34, 0.84), "upright"),
        ("room_02", "item_06", (0.62, 0.30, 0.80, 0.74), "upright"),
        ("room_05", "item_05", (0.40, 0.62, 0.66, 0.84), "upright"),
        ("room_08", "item_02", (0.55, 0.55, 0.72, 0.85), "upright"),
    ]
    for rn, itn, bx, mode in JOBS:
        room = Image.open(f"samples/rooms/{rn}.png").convert("RGB")
        item = Image.open(f"samples/items/{itn}.png").convert("RGB")
        W, H_ = room.size

        depth = estimate_depth(room)
        plane = floor_plane(room, depth)
        rgba = cutout(item, crop=True)

        box = (int(bx[0] * W), int(bx[1] * H_), int(bx[2] * W), int(bx[3] * H_))
        M = place_transform(plane, box, rgba.size, room.size, mode=mode)
        result = compose(room, rgba, M, plane)
        result.save(out_dir / f"{rn}__{itn}.png")

        s_near = scale_hint(plane, H_ * 0.90, room.size)
        s_far = scale_hint(plane, H_ * 0.55, room.size)
        print(f"  {rn} x {itn:<8} 지평선y={plane['horizon_y']:.0f}  "
              f"원근 크기비 가까이 {s_near:.2f} : 멀리 {s_far:.2f}")
    print(f"저장 위치: {out_dir}")
