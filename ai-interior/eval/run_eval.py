"""로컬 파이프라인 vs Gemini API 비교 실험 (보고서 핵심).

## 무엇을 비교하는가

주 비교 — **합성 방식**: 직접 구현한 기하 파이프라인 vs 상용 생성 API
부 비교 — **위치 지정 방식**: 마커 / 마스크 / 텍스트 (API 안에서만)

|          | 좌표 직접 | 마커 | 마스크 | 텍스트 |
|----------|----------|------|--------|--------|
| 로컬     |    O     |  -   |   -    |   X    |
| API      |    X     |  O   |   O    |   O    |

**로컬에 마커/마스크 칸이 없는 이유**: 로컬 파이프라인은 그림 위의 표시를 보지 않고
박스 좌표를 직접 받는다. 마커를 그리든 마스크를 칠하든 결과가 완전히 같으므로
같은 조건을 두 번 돌리는 셈이 된다.
**로컬에 텍스트 칸이 없는 이유**: 자연어 위치 지시를 해석하는 기능이 없다(좌표가 필수 입력).
이 비대칭 자체가 두 방식의 본질적 차이이고 보고서에 쓸 결과다.

## 지표

CLIP delta — 배치 영역 안쪽 품질. 실제로 우열이 갈리는 축.
마스크 밖 SSIM — 무결성 확인용이며 **우열 지표가 아니다**. 로컬은 알파 합성이라
정의상 1.0에 가깝다. 자세한 이유는 eval/metrics.py 상단 참조.
사람 설문 — scoresheet.csv에 위치정확도/자연스러움/가구보존을 1-5로 기입.

## 사용법

    python eval/run_eval.py                      # mock (요금 없음)
    REAL_API=1 python eval/run_eval.py --yes     # 10쌍, API 30회 (약 $2)
    REAL_API=1 python eval/run_eval.py --all --yes   # 100쌍, API 300회 (약 $20)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api_baseline import (COST_PER_CALL, MARKER_ALPHA, PROMPT_TEMPLATE,  # noqa: E402
                          REAL_API, draw_marker, draw_overlay, generate)
from eval.metrics import clip_delta, identical_ratio, ssim_outside  # noqa: E402
from pipeline.compose import compose  # noqa: E402
from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import floor_plane, place_transform  # noqa: E402
from pipeline.segment import cutout  # noqa: E402

RESULTS = ROOT / "eval" / "results"

with open(ROOT / "samples" / "items.json", encoding="utf-8") as _f:
    ITEMS = json.load(_f)["items"]          # 단일 출처. app.py도 같은 파일을 읽는다.

# 조건 4개. engine='api' 인 것만 요금이 발생한다.
CONDITIONS = [
    {"key": "local",      "engine": "local", "signal": "coords", "label": "로컬 (좌표)"},
    {"key": "api_marker", "engine": "api",   "signal": "marker", "label": "API (마커)"},
    {"key": "api_mask",   "engine": "api",   "signal": "mask",   "label": "API (마스크)"},
    {"key": "api_text",   "engine": "api",   "signal": "text",   "label": "API (텍스트)"},
]
PAID = [c for c in CONDITIONS if c["engine"] == "api"]

# 방마다 고정된 배치 영역 (x%, y%, w%, h%). 실험 재현을 위해 하드코딩한다.
BOXES = [
    (34, 58, 32, 30), (12, 60, 30, 28), (55, 57, 33, 31), (30, 62, 36, 26),
    (60, 60, 28, 30), (18, 58, 34, 32), (42, 61, 30, 28), (25, 59, 32, 30),
    (50, 62, 34, 26), (36, 57, 30, 33),
]


# --- 입력 만들기 -----------------------------------------------------------

def pixel_box(room: Image.Image, pct):
    """(x%, y%, w%, h%) 를 **코너 규약** (x0, y0, x1, y1) 픽셀로. app.py와 같은 규약."""
    W, H = room.size
    x0, y0 = int(W * pct[0] / 100), int(H * pct[1] / 100)
    return x0, y0, min(x0 + int(W * pct[2] / 100), W - 1), min(y0 + int(H * pct[3] / 100), H - 1)


def make_mask_input(room: Image.Image, box) -> Image.Image:
    """배치 영역을 반투명 흰색으로 덮는다.

    마커와 **같은 투명도**(MARKER_ALPHA)를 쓴다. 이전에는 불투명 흰색이라 그 아래
    바닥 텍스처를 지웠고, 마커는 반투명이라 원본이 비쳤다 — "위치 지시 문장만
    다르다"는 통제가 입력 이미지의 정보량에서 이미 깨져 있었다.
    이제 두 방식의 차이는 색(빨강/흰색)뿐이다.
    """
    return draw_overlay(room, box, (255, 255, 255, MARKER_ALPHA))


def where_phrase(room: Image.Image, box) -> str:
    """박스 위치를 자연어 지시로. 텍스트 조건에만 쓴다."""
    cx = (box[0] + box[2]) / 2 / room.width
    side = "왼쪽" if cx < 0.38 else ("오른쪽" if cx > 0.62 else "가운데")
    return f"{side} 바닥"


# --- 방식별 프롬프트 (위치 지시 문장만 다르다) -------------------------------

def _with_name(prompt: str, item_name: str) -> str:
    return f"{prompt}\n배치할 가구: {item_name}" if item_name else prompt


def prompt_for(signal: str, item_name: str, where: str) -> str:
    if signal == "marker":
        return _with_name(PROMPT_TEMPLATE, item_name)
    if signal == "mask":
        p = PROMPT_TEMPLATE.replace("반투명 사각형으로 표시된 위치", "흰색으로 표시된 영역")
        p = p.replace("표시용 사각형은 결과에 남기지 말 것.", "표시용 영역은 결과에 남기지 말 것.")
        return _with_name(p, item_name)
    p = PROMPT_TEMPLATE.replace("첫 번째 이미지에서 반투명 사각형으로 표시된 위치",
                                f"첫 번째 이미지의 {where}")
    return _with_name(p.replace("\n표시용 사각형은 결과에 남기지 말 것.", ""), item_name)


# --- 조건 하나 실행 --------------------------------------------------------

def run_condition(cond, room, item, box, meta, cache):
    """(입력 이미지, 결과 이미지, 프롬프트) 반환. 프롬프트는 로컬이면 None."""
    if cond["engine"] == "local":
        key = id(room)
        if key not in cache:
            cache[key] = floor_plane(room, estimate_depth(room))
        plane = cache[key]
        if plane["coef"] is None:
            raise RuntimeError("바닥 평면 추정 실패")
        rgba = cutout(item, crop=True)
        M = place_transform(plane, box, rgba.size, room.size, mode=meta.get("mode", "upright"))
        marked = draw_marker(room, box)          # 시각 확인용. 모델에는 안 들어간다.
        return marked, compose(room, rgba, M, plane), None

    where = where_phrase(room, box)
    src = {"marker": lambda: draw_marker(room, box),
           "mask": lambda: make_mask_input(room, box),
           "text": lambda: room}[cond["signal"]]()
    prompt = prompt_for(cond["signal"], meta["name"], where)
    return src, generate(src, item, prompt), prompt


# --- 비교 그리드 -----------------------------------------------------------

def build_grid(rows):
    """rows: [(pair_id, 원본, {조건키: 결과})] -> 원본 + 조건 4개 = 5열 이미지."""
    cell_w, pad, head = 340, 8, 26
    labels = ["ORIGINAL"] + [c["key"].upper() for c in CONDITIONS]
    cols = len(labels)

    def fit(im):
        return im.resize((cell_w, int(cell_w * im.height / im.width)))

    thumbs = [[fit(r[1])] + [fit(r[2][c["key"]]) for c in CONDITIONS] for r in rows]
    cell_h = max(im.height for row in thumbs for im in row)

    grid = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                             head + len(rows) * (cell_h + pad) + pad), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(labels):
        d.text((pad + c * (cell_w + pad) + 6, 8), label, fill=(40, 40, 40))
    for r, row in enumerate(thumbs):
        y = head + r * (cell_h + pad)
        d.text((4, y + 4), rows[r][0][:9], fill=(120, 120, 120))
        for c, im in enumerate(row):
            grid.paste(im, (pad + c * (cell_w + pad), y))
    return grid


def _next_run_name(root: Path, prefix: str) -> str:
    """real-001, real-002 … 다음 번호. 기존 결과를 절대 덮지 않는다."""
    n = 1
    while (root / f"{prefix}-{n:03d}").exists():
        n += 1
    return f"{prefix}-{n:03d}"


# --- 메인 ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="방 10 x 가구 10 = 100조합 (기본은 10조합)")
    ap.add_argument("--limit", type=int, default=0, help="조합 수 제한")
    ap.add_argument("--yes", action="store_true", help="REAL_API=1 일 때 실제 호출 확인")
    ap.add_argument("--run", default=None, help="결과 폴더 이름 (기본: real-NNN / mock-NNN)")
    ap.add_argument("--no-metrics", action="store_true", help="CLIP/SSIM 계산 생략 (빠른 확인용)")
    args = ap.parse_args()

    rooms = sorted((ROOT / "samples" / "rooms").glob("*.png"))[:10]
    items = sorted((ROOT / "samples" / "items").glob("*.png"))[:10]
    if not rooms or not items:
        sys.exit("samples/rooms, samples/items 에 이미지가 없습니다.")

    pairs = list(product(range(len(rooms)), range(len(items)))) if args.all \
        else [(i, i) for i in range(min(len(rooms), len(items)))]
    if args.limit:
        pairs = pairs[: args.limit]

    paid_calls = len(pairs) * len(PAID)
    print(f"조합 {len(pairs)}개 x 조건 {len(CONDITIONS)}개 = 결과 {len(pairs) * len(CONDITIONS)}장")
    print(f"그중 유료 API 호출 {paid_calls}회 "
          f"({'실제 — 약 $%.2f' % (paid_calls * COST_PER_CALL) if REAL_API else 'mock — 요금 없음'})")
    if REAL_API and not args.yes:
        sys.exit(f"REAL_API=1 입니다. 약 ${paid_calls * COST_PER_CALL:.2f} 가 발생합니다. "
                 "진행하려면 --yes 를 붙이세요.")

    # 폴더는 확인을 통과한 뒤에 만든다. 중단된 실행이 빈 폴더를 남기면 안 된다.
    out_root = RESULTS / (args.run or _next_run_name(RESULTS, "real" if REAL_API else "mock"))
    if out_root.exists() and not args.run:
        sys.exit(f"{out_root} 가 이미 있습니다. --run 으로 다른 이름을 주세요.")
    out_root.mkdir(parents=True, exist_ok=True)
    for c in CONDITIONS:
        (out_root / c["key"]).mkdir(exist_ok=True)
    print(f"저장 폴더: {out_root}\n")

    plane_cache, sheet, grid_rows, failures = {}, [], [], []
    for ri, ii in pairs:
        room = Image.open(rooms[ri]).convert("RGB")
        item = Image.open(items[ii]).convert("RGB")
        pair_id = f"{rooms[ri].stem}__{items[ii].stem}"
        box = pixel_box(room, BOXES[ri % len(BOXES)])
        meta = ITEMS[ii % len(ITEMS)]
        outs = {}

        for cond in CONDITIONS:
            tag = f"{pair_id} [{cond['key']}]"
            try:
                src, result, prompt = run_condition(cond, room, item, box, meta, plane_cache)
            except Exception as e:                      # 한 칸이 실패해도 나머지는 계속
                print(f"  {tag} 실패: {e}")
                failures.append((tag, str(e)))
                outs[cond["key"]] = room
                continue

            d = out_root / cond["key"]
            src.save(d / f"{pair_id}_input.png")
            result.save(d / f"{pair_id}_output.png")
            if prompt:
                (d / f"{pair_id}_prompt.txt").write_text(prompt, encoding="utf-8")
            outs[cond["key"]] = result

            row = {"pair_id": pair_id, "condition": cond["key"], "item": meta["name"],
                   "engine": cond["engine"], "signal": cond["signal"]}
            if not args.no_metrics:
                row.update(clip_delta(room, result, meta["en"]))
                row["ssim_outside"] = round(ssim_outside(room, result, box), 4)
                row["identical_ratio"] = round(identical_ratio(room, result), 4)
            sheet.append(row)
            print(f"  {tag}  " + (f"CLIP {row.get('clip_delta', 0):+.4f}"
                                  f"  SSIM_out {row.get('ssim_outside', 0):.4f}"
                                  if not args.no_metrics else "저장됨"))

        grid_rows.append((pair_id, room, outs))

    build_grid(grid_rows).save(out_root / "comparison_grid.png")

    cols = ["pair_id", "condition", "engine", "signal", "item",
            "clip_before", "clip_after", "clip_delta", "ssim_outside", "identical_ratio",
            "위치정확도(1-5)", "합성자연스러움(1-5)", "가구보존(1-5)"]
    with open(out_root / "scoresheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sheet)

    # 조건별 평균 — 보고서 표의 초안이 된다
    if sheet and not args.no_metrics:
        print("\n조건별 평균")
        print(f"  {'조건':<14}{'CLIP delta':>12}{'마스크밖 SSIM':>15}{'원본동일비율':>14}")
        for c in CONDITIONS:
            rs = [r for r in sheet if r["condition"] == c["key"]]
            if not rs:
                continue
            avg = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"  {c['label']:<14}{avg('clip_delta'):>+12.4f}"
                  f"{avg('ssim_outside'):>15.4f}{avg('identical_ratio') * 100:>13.1f}%")

    if failures:
        print(f"\n실패 {len(failures)}건:")
        for t, e in failures:
            print(f"  {t}: {e}")

    print(f"\n완료 → {out_root}")
    print("  comparison_grid.png : 원본 + 조건 4개 비교")
    print("  scoresheet.csv      : 정량 지표 + 사람 채점 3칸 (자동 채점 아님)")


if __name__ == "__main__":
    main()
