"""크기 자동 보정 검증 — ADE20K 정답의 문을 자로 쓴다 (DEVLOG §29).

크기 공식  화면 높이 = (접지 y − 지평선 y) × 실제 높이 / 카메라 높이  의 절대 크기를
한 번도 정답과 대조하지 않았다. 문은 높이가 거의 표준(2.03m, ±5%)이라 좋은 자다.

문 고르는 조건(결과를 보기 전에 정함):
  - ADE20K 정답에서 문(15번) 연결 요소, 면적 0.5% 이상
  - 위아래가 사진 끝에 닿지 않음 (전체가 보임)
  - 바로 아래 6px 안에 바닥(4번)이 30% 이상 (접지선이 가구에 가리지 않음)
  - 세로/가로 1.5~5 (옆으로 열려 얇게 보이는 문, 양문 제외)
  - 화면 높이 = 가운데 50% 열에서 잰 위·아래 끝의 중앙값

    python eval/size_check.py --limit 400

결과: eval/results/size_check/doors.csv (문마다 실측 높이, 지평선, 공식 예측).
짝수 번째 문은 방법을 고르는 데(dev), 홀수 번째는 최종 보고(test)에만 쓴다.

09-19: 처음 실행은 깊이 평면에서 외삽한 지평선을 썼다(horizon_depth_y) — test ±25% 안 6%.
dev에서 고른 고정 지평선(사진 높이 41%, geometry.HORIZON_FRAC)으로 바꾼 뒤의 수치는
horizon_y/ratio 열이다. dev는 그 값을 정하는 데 썼으므로 test만 성능으로 읽는다. DEVLOG §30.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import CAM_HEIGHT_M, floor_plane  # noqa: E402

ADE = Path.home() / "capstone-data" / "ADEChallengeData2016"
SCENES = {"bedroom", "living_room", "dining_room", "home_office", "hotel_room", "dorm_room"}
DOOR, FLOOR = 15, 4
DOOR_M = 2.03
OUT = ROOT / "eval" / "results" / "size_check"


def used_ids() -> set[str]:
    """이미 샘플로 쓴 방(튜닝셋·홀드아웃)은 뺀다."""
    txt = (ROOT / "samples" / "SOURCES.md").read_text(encoding="utf-8")
    import re
    return set(re.findall(r"ADE_(?:train|val)_\d{8}", txt))


def doors_in(ann: np.ndarray):
    H, W = ann.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((ann == DOOR).astype(np.uint8), 8)
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < 0.005 * H * W or y <= 2 or y + h >= H - 2:
            continue
        if not (1.5 <= h / max(w, 1) <= 5.0):
            continue
        m = lab == k
        cols = range(x + w // 4, x + 3 * w // 4 + 1)
        tops, bots = [], []
        for c in cols:
            ys = np.where(m[:, c])[0]
            if len(ys):
                tops.append(ys.min()); bots.append(ys.max())
        if len(bots) < 3:
            continue
        top, bot = float(np.median(tops)), float(np.median(bots))
        below = ann[int(bot) + 1:min(int(bot) + 7, H), x + w // 4:x + 3 * w // 4 + 1]
        if below.size == 0 or (below == FLOOR).mean() < 0.3:
            continue
        yield {"x": int(x + w / 2), "top": top, "bottom": bot, "h_px": bot - top}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="문 개수 상한 (깊이 추정 시간)")
    args = ap.parse_args()

    scenes = {}
    for line in (ADE / "sceneCategories.txt").read_text().splitlines():
        iid, sc = line.split()
        if sc in SCENES:
            scenes[iid] = sc
    skip = used_ids()
    OUT.mkdir(parents=True, exist_ok=True)

    cands = []
    for iid in sorted(scenes):
        if iid in skip:
            continue
        split = "training" if "_train_" in iid else "validation"
        ann = np.array(Image.open(ADE / "annotations" / split / f"{iid}.png"))
        if ann.shape[1] <= ann.shape[0]:           # 가로 사진만 (샘플과 같은 조건)
            continue
        for d in doors_in(ann):
            cands.append((iid, split, d))
        if len(cands) >= args.limit:
            break
    print(f"문 {len(cands)}개 (방 {len({c[0] for c in cands})}장)")

    rows, depth_cache = [], {}
    for i, (iid, split, d) in enumerate(cands):
        if iid not in depth_cache:
            room = Image.open(ADE / "images" / split / f"{iid}.jpg").convert("RGB")
            depth = estimate_depth(room)
            depth_cache = {iid: (room, floor_plane(room, depth))}
        room, plane = depth_cache[iid]
        H = room.size[1]
        hy = plane.get("horizon_y")
        hd = plane.get("horizon_depth_y")
        pred = (d["bottom"] - hy) * DOOR_M / CAM_HEIGHT_M if hy is not None and d["bottom"] > hy else None
        rows.append({"idx": i, "split": "dev" if i % 2 == 0 else "test", "image": iid,
                     "scene": scenes[iid], "H": H, "door_x": d["x"], "door_top": round(d["top"], 1),
                     "door_bottom": round(d["bottom"], 1), "door_h_px": round(d["h_px"], 1),
                     "horizon_y": None if hy is None else round(hy, 1),
                     "horizon_depth_y": None if hd is None else round(hd, 1),
                     "pred_h_px": None if pred is None else round(pred, 1),
                     "ratio": None if pred is None else round(d["h_px"] / pred, 3)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cands)}", flush=True)

    with open(OUT / "doors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for sp in ("dev", "test"):
        r = np.array([x["ratio"] for x in rows if x["split"] == sp and x["ratio"] is not None])
        fail = sum(1 for x in rows if x["split"] == sp and x["ratio"] is None)
        print(f"{sp}: n={len(r)} (지평선 없음/접지 위 {fail})  실측/공식 중앙값 {np.median(r):.2f}  "
              f"사분위 {np.percentile(r, 25):.2f}~{np.percentile(r, 75):.2f}  "
              f"±25% 안 {np.mean(np.abs(r - 1) <= 0.25) * 100:.0f}%")
    print(f"저장: {OUT / 'doors.csv'}")


if __name__ == "__main__":
    main()
