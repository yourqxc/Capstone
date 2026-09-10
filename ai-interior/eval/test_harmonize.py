"""조명 정합의 계약 검증 — "흰 가구는 갈변하지 않는다".

이전 구현은 가구 색을 **바닥** 평균 쪽으로 끌어당겨서, 짙은 마루 방에서 흰 가구가
누렇게 변했다. 사용자의 회베이지 침대에서는 **보정을 아예 안 한 것보다 나쁜** 결과가 나왔다.
이 파일은 그 회귀를 코드로 못박는다.

    python eval/test_harmonize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.compose import harmonize, illum_field  # noqa: E402
from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import floor_plane  # noqa: E402

PATCHES = {"백": (250, 250, 250), "중회": (128, 128, 128), "흑": (30, 30, 30),
           "적": (200, 60, 50), "청": (60, 90, 180), "목재": (150, 110, 70)}


# 중성색 채도 상한의 여유값. 실측으로 정했다 — room_08의 중회 패치가 4.11로 가장 빡빡하고
# 방 백색점 5.6 x 0.35 = 1.96 이므로 여유 2.5가 필요하다. 이 값을 넘으면 갈변으로 본다.
CHROMA_MARGIN = 2.5
HUE_LIMIT_DEG = 8.0     # 실측 최대 7.5도(room_04 목재). 사람 눈에 보이는 경계가 5~10도다.


def chroma(rgb, m):
    lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    return float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)[m].mean())


def hue_deg(rgb, m):
    lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    a, b = lab[..., 1][m].mean() - 128, lab[..., 2][m].mean() - 128
    return float(np.degrees(np.arctan2(b, a)))


def main():
    rooms = sorted((ROOT / "samples" / "rooms").glob("*.png"))
    amount, fails = 0.35, []
    print(f"방 {len(rooms)}개 x 패치 {len(PATCHES)}종, amount={amount}\n")
    print(f"{'방':<10}{'방 백색점':>10}" + "".join(f"{k:>7}" for k in PATCHES)
          + "   최대색상각")

    for rp in rooms:
        room = Image.open(rp).convert("RGB")
        W, H = room.size
        d = estimate_depth(room)
        plane = floor_plane(room, d)
        field = illum_field(room, plane)
        w_room = float(np.hypot(*field["w_room"]))

        alpha = np.zeros((H, W), np.float32)
        alpha[int(H * 0.60):int(H * 0.90), int(W * 0.36):int(W * 0.64)] = 1.0
        m = alpha > 0.6

        cs, hues = [], []
        limit = w_room * amount + CHROMA_MARGIN
        for name, col in PATCHES.items():
            rgb = np.zeros((H, W, 3), np.float32)
            rgb[m] = col
            out = harmonize(rgb, alpha, room, plane["mask"], amount, field=field, depth=d)
            c = chroma(out, m)
            cs.append(c)
            # 계약 1 — 중성색은 갈변하지 않는다. 채도가 방 백색점 x amount + 여유를 넘지 않을 것.
            if name in ("백", "중회", "흑") and c > limit:
                fails.append(f"{rp.stem} {name} 패치 채도 {c:.2f} > 상한 {limit:.2f}")
            # 계약 2 — 유채색 가구의 색상은 유지된다. 조명 정합이지 색 변환이 아니다.
            if name in ("적", "청", "목재"):
                dh = abs(((hue_deg(out, m) - hue_deg(rgb, m)) + 180) % 360 - 180)
                hues.append(dh)
                if dh > HUE_LIMIT_DEG:
                    fails.append(f"{rp.stem} {name} 색상각 {dh:.1f}도 > 상한 {HUE_LIMIT_DEG}도")
        print(f"{rp.stem:<10}{w_room:10.1f}" + "".join(f"{c:7.2f}" for c in cs)
              + f"{max(hues) if hues else 0:12.1f}")

    print()
    if fails:
        print(f"실패 {len(fails)}건")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("통과 — 중성색이 갈변하지 않고 유채색의 색상이 유지된다")


if __name__ == "__main__":
    main()
