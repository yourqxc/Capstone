"""유료 파일럿(09-09, API 9회 약 $0.60) 재채점 — API를 다시 부르지 않는다 (DEVLOG §28).

파일럿은 CLIP을 **이미지 전체**로 채점했다. 가구가 화면의 몇 %라 구별력이 없다는 것이
§19·§23에서 확인됐으므로 배치 영역 CLIP으로 다시 잰다. 저장된 Gemini 결과 이미지만 쓴다.

로컬 쪽은 파일럿 당시 결과(누끼가 깨진 기본 박스, 깊이 없는 합성)가 지금 파이프라인과
다르므로 **같은 배치 박스로 지금 파이프라인을 다시 돌려** 비교한다. 비용은 없다.

    python eval/rescore_pilot.py

CLIP은 같은 방·같은 가구 안에서 조건끼리 비교할 때만 의미가 있다(§23). 그래서 평균보다
쌍마다 어느 조건이 높은지를 본다.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import clip_delta, crop_around  # noqa: E402
from eval.run_eval import ITEMS, pixel_box  # noqa: E402
from pipeline.compose import _warp_rgba, compose  # noqa: E402
from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import auto_height_px, floor_plane, place_transform  # noqa: E402
from pipeline.segment import cutout, default_box, pct_box  # noqa: E402

PILOT = ROOT / "eval" / "results" / "pilot"
OUT = PILOT / "rescore"
# 파일럿 당시 run_eval의 배치 박스. 이후 room_01은 (34, 60)으로 옮겼지만(§19) Gemini가 받은 입력은 이 박스다.
PILOT_BOXES = {1: (34, 58, 32, 30), 2: (12, 60, 30, 28), 3: (55, 57, 33, 31)}
API = [("api_marker", "Gemini (마커)"), ("api_mask", "Gemini (마스크)"), ("api_text", "Gemini (텍스트)")]


def local_now(room, item, meta, box, painted, refine_strength=None):
    depth = estimate_depth(room)
    plane = floor_plane(room, depth)
    ib = pct_box(item.size, meta["box"]) if painted else default_box(item.size)
    rgba = cutout(item, box=ib, crop=True)
    M = place_transform(plane, box, rgba.size, room.size,
                        height_px=auto_height_px(plane, box[3], room.size, meta["height_m"]))
    out = compose(room, rgba, M, plane, depth=depth)
    if refine_strength:
        from pipeline.refine import refine
        out = refine(out, _warp_rgba(rgba, M, room.size)[1], box, meta["en"], refine_strength)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    try:
        import diffusers  # noqa: F401
        with_refine = True
    except ImportError:
        with_refine = False
    conds = [("pilot_local", "로컬 (09-09 당시)"), ("local_auto", "로컬 (자동)"), ("local_box", "로컬 (가구 박스)")]
    if with_refine:
        conds.append(("local_box_refine", "로컬 (가구 박스 + 다듬기 0.35)"))
    conds += API

    rows, crops = [], []
    for i, pct in PILOT_BOXES.items():
        pid = f"room_{i:02d}__item_{i:02d}"
        room = Image.open(ROOT / f"samples/rooms/room_{i:02d}.png").convert("RGB")
        item = Image.open(ROOT / f"samples/items/item_{i:02d}.png").convert("RGB")
        meta, box = ITEMS[i - 1], pixel_box(room, pct)
        outs = {"pilot_local": Image.open(PILOT / "local" / f"{pid}_output.png").convert("RGB"),
                "local_auto": local_now(room, item, meta, box, painted=False),
                "local_box": local_now(room, item, meta, box, painted=True)}
        if with_refine:
            outs["local_box_refine"] = local_now(room, item, meta, box, painted=True, refine_strength=0.35)
        for key, _ in API:
            outs[key] = Image.open(PILOT / key / f"{pid}_output.png").convert("RGB")
        for key, label in conds:
            res = outs[key].resize(room.size) if outs[key].size != room.size else outs[key]
            if key.startswith("local"):
                res.save(OUT / f"{pid}_{key}.png")
            rows.append({"pair_id": pid, "condition": key, "label": label,
                         "clip_crop": clip_delta(room, res, meta["en"], box)["clip_delta"],
                         "clip_full": clip_delta(room, res, meta["en"])["clip_delta"]})
        crops.append((pid, [crop_around(outs[k].resize(room.size), box) for k, _ in conds]))
        print(f"  {pid}: " + "  ".join(f"{k}={r['clip_crop']:+.4f}" for (k, _), r in zip(conds, rows[-len(conds):])))

    with open(OUT / "rescore.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'조건':<32}{'영역 CLIP 평균':>14}{'전체 CLIP 평균':>14}")
    for key, label in conds:
        rs = [r for r in rows if r["condition"] == key]
        print(f"{label:<32}{sum(r['clip_crop'] for r in rs) / len(rs):>+14.4f}"
              f"{sum(r['clip_full'] for r in rs) / len(rs):>+14.4f}")

    # 쌍마다 1등 조건
    print("\n쌍별 영역 CLIP 1등:")
    for pid, _ in crops:
        rs = sorted((r for r in rows if r["pair_id"] == pid), key=lambda r: -r["clip_crop"])
        print(f"  {pid}: " + " > ".join(f"{r['label']}" for r in rs[:3]))

    # 비교 그림
    font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 15)
    cell = 230
    sheet = Image.new("RGB", (len(conds) * (cell + 6) + 6, len(crops) * (cell + 26) + 30), "white")
    d = ImageDraw.Draw(sheet)
    for c, (_, label) in enumerate(conds):
        d.text((6 + c * (cell + 6), 6), label, fill=(0, 0, 0), font=font)
    for r, (pid, ims) in enumerate(crops):
        for c, im in enumerate(ims):
            t = im.resize((cell, int(cell * im.height / im.width)))
            sheet.paste(t, (6 + c * (cell + 6), 30 + r * (cell + 26)))
            v = next(x["clip_crop"] for x in rows if x["pair_id"] == pid and x["condition"] == conds[c][0])
            d.text((8 + c * (cell + 6), 30 + r * (cell + 26) + t.height + 2), f"{pid[:7]}  CLIP {v:+.4f}",
                   fill=(60, 60, 60), font=font)
    sheet.save(OUT / "rescore_grid.png")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
