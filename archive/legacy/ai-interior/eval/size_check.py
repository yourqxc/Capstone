"""크기 공식의 대리 검증 — ADE20K 문 영역과 가정한 치수를 비교한다.

크기 공식  화면 높이 = (접지 y − 지평선 y) × 실제 높이 / 카메라 높이  의 절대 크기를
점검한다. 문 높이 2.03m와 카메라 높이 1.4m는 측정된 정답이 아닌 가정이다.

문 고르는 조건(결과를 보기 전에 정함):
  - ADE20K 정답에서 문(15번) 연결 요소, 면적 0.5% 이상
  - 위아래가 사진 끝에 닿지 않음 (전체가 보임)
  - 바로 아래 6px 안에 바닥(4번)이 30% 이상 (접지선이 가구에 가리지 않음)
  - 세로/가로 1.5~5 (옆으로 열려 얇게 보이는 문, 양문 제외)
  - 화면 높이 = 가운데 50% 열에서 잰 위·아래 끝의 중앙값

    python eval/size_check.py --limit 400

결과: eval/results/size_check/doors_by_room.csv. 사진 ID로 dev/test를 나누므로
같은 사진의 여러 문은 반드시 같은 그룹에 속한다. 기존 doors.csv는 09-19의 문별 분할
기록으로 보존한다. 기존 사진을 재평가한 수치는 새로운 미관측 사진의 검증이 아니다.
사진 높이 41%라는 상수는 기존 데이터로 결정됐다는 한계도 함께 해석해야 한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def evaluation_split(image_id: str) -> str:
    """문 개수나 탐색 순서가 달라도 한 방은 한 그룹에만 배정한다."""
    return "dev" if hashlib.sha256(image_id.encode()).digest()[0] % 2 == 0 else "test"


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
    if args.limit < 1:
        ap.error("--limit은 1 이상이어야 합니다.")

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
    if not cands:
        sys.exit("조건에 맞는 문이 없습니다.")

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
        rows.append({"idx": i, "split": evaluation_split(iid), "image": iid,
                     "scene": scenes[iid], "H": H, "door_x": d["x"], "door_top": round(d["top"], 1),
                     "door_bottom": round(d["bottom"], 1), "door_h_px": round(d["h_px"], 1),
                     "horizon_y": None if hy is None else round(hy, 1),
                     "horizon_depth_y": None if hd is None else round(hd, 1),
                     "pred_h_px": None if pred is None else round(pred, 1),
                     "ratio": None if pred is None else round(d["h_px"] / pred, 3)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cands)}", flush=True)

    dest = OUT / "doors_by_room.csv"
    with open(dest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for sp in ("dev", "test"):
        r = np.array([x["ratio"] for x in rows if x["split"] == sp and x["ratio"] is not None])
        fail = sum(1 for x in rows if x["split"] == sp and x["ratio"] is None)
        total = len(r) + fail
        passed = int(np.count_nonzero(np.abs(r - 1) <= 0.25))
        print(f"{sp}: 전체 {total}, 계산 성공 {len(r)}, 계산 실패 {fail}, ±25% 안 {passed}")
        if len(r):
            print(f"  계산 성공 중 {passed}/{len(r)} ({passed / len(r):.1%}), "
                  f"전체 중 {passed}/{total} ({passed / total:.1%}), "
                  f"관측/예측 중앙값 {np.median(r):.2f}")
    print("주의: 문·카메라 치수는 가정이며, 기존 데이터의 방 단위 재분할은 새 홀드아웃이 아닙니다.")
    print(f"저장: {dest}")


if __name__ == "__main__":
    main()
