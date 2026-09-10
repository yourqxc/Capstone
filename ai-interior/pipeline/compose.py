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


def _lab(rgb):
    """float LAB. 8bit 양자화를 거치지 않는다 — 이동량이 수 단위라 반올림이 크게 먹는다."""
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _to_rgb(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1) * 255.0


def _spread(v):
    """로버스트 스프레드(표준편차 대용). 이상치에 흔들리지 않는다."""
    return 0.5 * (np.percentile(v, 84) - np.percentile(v, 16))


def illum_field(room: Image.Image, plane: dict | None = None) -> dict:
    """방에서 **알베도가 아니라 조명만** 뽑는다. 방 1장당 1회 계산하면 된다.

    핵심은 참조를 어디서 가져오느냐다. 이전 구현은 **바닥**의 평균색을 참조로 썼는데,
    그러면 짙은 마루 방에서 흰 가구가 마루색으로 물든다(갈변). 실측으로 확인된 문제다 —
    순백 가구를 놓았을 때 채도가 평균 3.77까지 올랐고, 사용자의 회베이지 침대에서는
    **보정을 아예 안 한 것(5.79)보다 나쁜 6.73**이 나왔다.

    그래서 백색점은 바닥이 아니라 **방의 밝은 띠(P80~P99)**에서 가져온다. 밝은 표면일수록
    알베도가 1에 가까워 조명색에 근접하기 때문이다. 창 과노출(L>=97)은 제외한다.
    """
    lab = _lab(np.asarray(room.convert("RGB")))
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    H, W = L.shape
    floor = plane["mask"] if (plane and plane.get("mask") is not None) else np.zeros(L.shape, bool)
    Ls = L[::4, ::4]                       # 백분위는 1/16 표본으로 충분하다

    band = (L >= np.percentile(Ls, 80)) & (L <= np.percentile(Ls, 99)) & (L < 97)
    if band.sum() < 500:
        band = L >= np.percentile(Ls, 80)
    w_room = np.array([np.median(A[band]), np.median(B[band])], np.float32)
    ok = Ls[Ls < 97]
    ok = ok if ok.size > 100 else Ls
    L_hi, L_sp = float(np.percentile(ok, 92)), float(_spread(ok))

    # 바닥 위 L의 1차식 기울기 = 방의 조명 방향. 같은 재질 위에서 재므로
    # 바닥 알베도는 상수항으로 흡수되고 기울기에는 조명만 남는다.
    p = q = 0.0
    sel = floor if floor.sum() > 500 else np.ones(L.shape, bool)
    floor_lab = (float(np.median(L[sel])), float(np.median(A[sel])), float(np.median(B[sel])))
    if floor.sum() > 2000:
        ys, xs = np.nonzero(floor)
        step = max(1, len(ys) // 20000)
        ys, xs = ys[::step], xs[::step]
        M = np.stack([xs / W, ys / H, np.ones(len(xs))], 1).astype(np.float32)
        d = L[ys, xs]
        coef = np.linalg.lstsq(M, d, rcond=None)[0]
        for _ in range(2):                 # IRLS 2회 — 러그·그림자·바닥 오분류를 억제
            r = d - M @ coef
            w = 1.0 / (1.0 + (r / max(1.0, 1.5 * np.median(np.abs(r)))) ** 2)
            coef = np.linalg.lstsq(M * w[:, None], d * w, rcond=None)[0]
        p, q = float(coef[0]), float(coef[1])

    return {"L": L, "A": A, "B": B, "floor": floor, "w_room": w_room,
            "L_hi": L_hi, "L_sp": L_sp, "floor_lab": floor_lab, "grad": (p, q),
            "light": float(np.clip(-p / 12.0, -1, 1))}


def _ring(alpha, floor, depth=None, frac=0.6):
    """가구 실루엣 주변의 바닥 링. 접지점과 깊이가 비슷한 부분만 남긴다.

    "이 자리가 방 평균보다 어두운가 밝은가"를 재기 위한 국소 참조다.
    링과 바닥 전체의 **비**를 쓰므로 바닥 알베도가 약분되고 조명만 남는다.
    """
    m = (alpha > 0.5).astype(np.uint8)
    if m.sum() < 50 or not floor.any():
        return np.zeros(alpha.shape, bool)
    ys, xs = np.nonzero(m)
    rad = float(max(24, frac * max(np.ptp(ys) + 1, np.ptp(xs) + 1)))
    dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    ring = (dist > 2) & (dist < rad) & floor
    if depth is not None and ring.sum() > 300:
        d0 = float(depth[min(int(ys.max()), depth.shape[0] - 1), int(np.median(xs))])
        tol = max(1e-3, 0.5 * (np.percentile(depth[floor], 90) - np.percentile(depth[floor], 10)))
        near = ring & (np.abs(depth - d0) < tol)
        if near.sum() > 300:
            ring = near
    return ring


def harmonize(rgb, alpha, room: Image.Image, region=None, amount: float = 0.35,
              field: dict | None = None, depth=None, chroma_cap: float = 1.15,
              dir_amount: float = 1.0, ao: float = 0.12, sigma_amount: float = 1.0):
    """가구를 방 조명에 맞춘다. amount=0 이면 아무것도 하지 않는다(되돌아갈 지점).

    이전 구현은 가구의 LAB 평균을 **바닥** 평균 쪽으로 끌어당겼다. 그 결과 조명이 아니라
    바닥 재질색이 옮겨붙어 흰 가구가 누렇게 변했다. 여기서는 세 가지를 바꿨다.

    1. 참조를 바닥이 아니라 **방의 밝은 띠**로 (조명색에 가깝다)
    2. 명도를 덧셈이 아니라 **곱셈 게인**으로 (상대 대비가 남아 형태감이 유지된다)
    3. 색도는 **평행이동만** 하고 표준편차 정합은 하지 않으며, 마지막에 채도 캡을 건다

    실측(순백 가구, 방 10개): 평균 채도 3.77 → 1.90.
    """
    if amount <= 0:
        return rgb
    inside = alpha > 0.6
    if inside.sum() < 50:
        return rgb
    f = field or illum_field(room, {"mask": region} if region is not None else None)
    lab = _lab(rgb)
    Li, Ai, Bi = lab[..., 0], lab[..., 1], lab[..., 2]

    # 가구의 백색점 = 가구의 밝은 띠 색도 = 촬영 당시 조명
    li = Li[inside]
    hi = inside & (Li >= np.percentile(li, 85)) & (Li <= np.percentile(li, 99))
    if hi.sum() < 30:
        hi = inside
    w_item = np.array([np.median(Ai[hi]), np.median(Bi[hi])], np.float32)
    L_hi_i, L_bar = float(np.percentile(li, 92)), float(np.median(li))

    # 노출 게인 — 하이라이트끼리 비교한다(평균끼리 비교하면 알베도가 섞인다)
    g_key = float(np.clip(f["L_hi"] / max(L_hi_i, 1.0), 0.75, 1.20))

    # 국소 셰이딩 — 링 vs 바닥 전체의 비. 같은 재질이라 바닥 알베도가 약분된다.
    R = _ring(alpha, f["floor"], depth)
    Lf, Af, Bf = f["floor_lab"]
    if R.sum() > 300:
        lr = f["L"][R]
        g_sh = float(np.clip(np.median(lr) / max(Lf, 1.0), 0.80, 1.25))
        d_loc = np.clip(np.array([np.median(f["A"][R]) - Af, np.median(f["B"][R]) - Bf],
                                 np.float32), -6, 6)
        # 링이 잡스러우면(벽·러그·다른 가구 혼입) 신뢰도가 0으로 떨어져 국소 항이 꺼진다
        trust = float(np.clip(1.5 - np.median(np.abs(lr - np.median(lr))) / 12.0, 0.0, 1.0))
    else:
        g_sh, d_loc, trust = 1.0, np.zeros(2, np.float32), 0.0
    g = 1.0 + amount * (g_key * (1 + trust * (g_sh - 1)) - 1.0)

    # 색도는 평행이동만. 표준편차 정합은 갈변의 기계장치라 걸지 않는다.
    d_ab = np.clip(amount * ((f["w_room"] - w_item) + trust * d_loc), -10, 10)
    t = np.clip(Li / 18.0, 0, 1) * np.clip((100.0 - Li) / 8.0, 0, 1)   # 흑/백 극단은 감쇠
    A2, B2 = Ai + d_ab[0] * t, Bi + d_ab[1] * t

    # 채도 캡 — 백색점은 방 쪽으로 옮기되 백색점 대비 채도는 키우지 않는다
    wn = w_item + d_ab
    c0 = np.hypot(Ai - w_item[0], Bi - w_item[1])
    c1 = np.hypot(A2 - wn[0], B2 - wn[1])
    scale = np.minimum(1.0, (c0 * chroma_cap + 1.5) / np.maximum(c1, 1e-3))
    A2, B2 = wn[0] + (A2 - wn[0]) * scale, wn[1] + (B2 - wn[1]) * scale

    # 명도 = 곱셈 게인 x 방향성 기울기 x 접지 차폐
    ys, xs = np.nonzero(inside)
    x0, x1, y_bot = xs.min(), xs.max(), ys.max()
    h = max(np.ptp(ys), 1)
    cx, half = (x0 + x1) / 2.0, max((x1 - x0) / 2.0, 1.0)
    xx = np.arange(lab.shape[1], dtype=np.float32)[None, :]
    yy = np.arange(lab.shape[0], dtype=np.float32)[:, None]
    k = float(np.clip(f["grad"][0] * (2 * half / lab.shape[1]) / max(L_bar, 1.0), -0.30, 0.30))
    g_dir = 1.0 + amount * dir_amount * 0.5 * k * np.clip((xx - cx) / half, -1, 1)
    g_ao = 1.0 - amount * ao * np.exp(-np.maximum(y_bot - yy, 0) / (0.10 * h))
    gam = float(np.clip(f["L_sp"] / max(_spread(li), 1e-3), 0.85, 1.18))
    gam = 1.0 + amount * sigma_amount * (gam - 1.0)
    Lm = L_bar * g
    L2 = Lm + (Li * g * g_dir * g_ao - Lm) * gam
    return _to_rgb(np.dstack([np.clip(L2, 0, 100), A2, B2])).astype(np.float32)


def compose(room: Image.Image, item_rgba: Image.Image, H: np.ndarray,
            plane: dict | None = None, shadow: bool = True,
            harmonize_amount: float = 0.35, depth=None) -> Image.Image:
    """가구를 방에 합성한다. H는 geometry.place_transform이 만든 호모그래피."""
    base = np.array(room.convert("RGB")).astype(np.float32)
    rgb, alpha = _warp_rgba(item_rgba, H, room.size)

    floor = plane.get("mask") if plane is not None else None
    field = illum_field(room, plane) if harmonize_amount > 0 else None
    if field is not None:
        rgb = harmonize(rgb, alpha, room, floor, harmonize_amount, field=field, depth=depth)

    if shadow:
        # 조명 방향을 harmonize와 같은 값으로 쓴다 (예전에는 서로 다른 추정을 썼다)
        light = field["light"] if field is not None else light_direction(room)
        # 그림자의 세로 폭도 바닥 평면에서 계산한다. 상수 0.12를 쓰면
        # 가구가 멀리 놓일수록 그림자가 실제보다 두 배 깊어진다.
        squash = None
        ys, _ = np.where(alpha > 0.5)
        if plane is not None and len(ys):
            from pipeline.geometry import perspective_squash

            squash = perspective_squash(plane, float(ys.max()),
                                        float(ys.max() - ys.min()), room.size[0])
        sh = (contact_shadow(alpha, squash=squash, light=light) if squash
              else contact_shadow(alpha, light=light))
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
