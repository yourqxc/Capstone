"""촬영 각도 불일치가 합성 품질을 언제부터 망가뜨리는가 (보고서용 실험).

## 왜 이 실험이 필요한가

2D 컷아웃 합성은 가구 사진에 **보이는 면만** 있다. 안 보이는 면은 유추할 수 없다.
따라서 방과 가구의 촬영 각도가 다르면 원리적으로 어긋난다.
그런데 실제로는 대부분 견딘다 — 실제 방에서 가구는 벽과 나란히만 놓이지 않으므로,
살짝 틀어진 가구는 "그렇게 놓인 것"으로 읽히기 때문이다.

**그래서 질문은 "어긋나는가"가 아니라 "얼마나 어긋나면 티가 나는가"다.**

## 방법

가구를 y축(수직축) 기준으로 회전시킬 수는 없다(3D 정보가 없다). 대신 **원근 왜곡을
단계적으로 가해** 촬영 각도가 다른 상황을 근사한다. 상단 폭을 줄이면 위에서 내려다본
것처럼, 늘리면 아래에서 올려다본 것처럼 보인다.

각 왜곡 단계마다 CLIP score를 재서 언제 급락하는지 찾는다.
CLIP은 "그 가구가 놓인 방"이라는 문장과의 부합도를 재므로, 가구가 어색해지면 떨어진다.

단독 실행:
    python eval/angle_limit.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import clip_score, placement_text  # noqa: E402
from pipeline.compose import compose  # noqa: E402
from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import floor_plane, place_transform  # noqa: E402
from pipeline.segment import cutout  # noqa: E402

# 상단 폭 배율. 1.0 = 원본, <1 = 위에서 내려다본 것처럼, >1 = 아래에서 올려다본 것처럼
SKEWS = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45]


def skew_topdown(rgba: Image.Image, top_scale: float) -> Image.Image:
    """상단 폭만 바꿔 촬영 고도가 다른 것처럼 만든다.

    수직축 회전이 아니라 원근 왜곡이다. 3D 정보가 없으므로 진짜 회전은 불가능하고,
    이것은 "카메라 고도가 달랐다면 생겼을 사다리꼴 왜곡"의 근사다.
    """
    if abs(top_scale - 1.0) < 1e-3:
        return rgba
    w, h = rgba.size
    dx = w * (1 - top_scale) / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(np.array(rgba.convert("RGBA")), M, (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    return Image.fromarray(arr)


def main():
    import json
    with open(ROOT / "samples" / "items.json", encoding="utf-8") as f:
        items = json.load(f)["items"]

    rooms = sorted((ROOT / "samples" / "rooms").glob("*.png"))[:5]
    out = ROOT / "eval" / "results" / "angle_limit"
    out.mkdir(parents=True, exist_ok=True)

    print(f"방 {len(rooms)}개 x 왜곡 {len(SKEWS)}단계 = {len(rooms) * len(SKEWS)}회 (전부 로컬, 요금 없음)\n")
    rows, panels = [], []

    for ri, rp in enumerate(rooms):
        room = Image.open(rp).convert("RGB")
        W, H = room.size
        meta = items[ri % len(items)]
        item = Image.open(ROOT / "samples" / "items" / meta["file"]).convert("RGB")
        plane = floor_plane(room, estimate_depth(room))
        if plane["coef"] is None:
            print(f"  {rp.stem} 바닥 추정 실패 — 건너뜀")
            continue
        base = cutout(item, crop=True)
        box = (int(W * 0.34), int(H * 0.56), int(W * 0.66), int(H * 0.92))
        text = placement_text(meta["en"])
        before = clip_score(room, text)

        line = []
        for s in SKEWS:
            warped = skew_topdown(base, s)
            M = place_transform(plane, box, warped.size, room.size)
            res = compose(room, warped, M, plane)
            score = clip_score(res, text)
            rows.append({"room": rp.stem, "item": meta["name"], "skew": s,
                         "clip_before": round(before, 4), "clip_after": round(score, 4),
                         "clip_delta": round(score - before, 4)})
            line.append(res)
            if ri == 0:
                res.save(out / f"skew_{s:.2f}.png")
        panels.append((rp.stem, meta["name"], line))
        print(f"  {rp.stem} x {meta['name']}: " +
              "  ".join(f"{s:.2f}→{rows[-len(SKEWS)+i]['clip_delta']:+.4f}" for i, s in enumerate(SKEWS)))

    # 왜곡 단계별 평균
    print(f"\n{'상단 폭 배율':>12}{'평균 CLIP delta':>18}  해석")
    best = None
    for s in SKEWS:
        rs = [r for r in rows if r["skew"] == s]
        if not rs:
            continue
        avg = sum(r["clip_delta"] for r in rs) / len(rs)
        if best is None or avg > best[1]:
            best = (s, avg)
        bar = "#" * max(int(avg * 400), 0)
        print(f"{s:>12.2f}{avg:>+18.4f}  {bar}")
    if best:
        print(f"\n최고점: 상단 폭 배율 {best[0]:.2f} (delta {best[1]:+.4f})")
        ref = [r["clip_delta"] for r in rows if r["skew"] == 1.0]
        base_avg = sum(ref) / len(ref) if ref else 0.0
        print(f"원본(1.00) 평균 {base_avg:+.4f} · 최고점과의 차이 {best[1] - base_avg:+.4f}")
        worst = min(((s, sum(r["clip_delta"] for r in rows if r["skew"] == s)
                      / max(len([r for r in rows if r["skew"] == s]), 1)) for s in SKEWS),
                    key=lambda t: t[1])
        drop = (best[1] - worst[1]) / best[1] * 100 if best[1] else 0
        print(f"최저점: 배율 {worst[0]:.2f} (delta {worst[1]:+.4f}) — 최고점 대비 {drop:.0f}% 하락")

    with open(out / "angle_limit.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 시각 비교 (첫 방)
    if panels:
        name, item_name, imgs = panels[0]
        cell = 300
        sheet = Image.new("RGB", (len(imgs) * cell, int(cell * imgs[0].height / imgs[0].width) + 22),
                          (248, 248, 248))
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        for i, (s, im) in enumerate(zip(SKEWS, imgs)):
            d.text((i * cell + 6, 4), f"top x{s:.2f}", fill=(20, 20, 20))
            r = im.copy(); r.thumbnail((cell - 4, cell))
            sheet.paste(r, (i * cell, 18))
        sheet.save(out / "angle_limit_grid.png")
    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
