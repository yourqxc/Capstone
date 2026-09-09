"""위치 지정 방식 3가지 비교 실험 (4주차 · 보고서 핵심).

고정된 방 사진 x 가구 사진 세트에 대해
  1) marker : 반투명 사각형 마커
  2) mask   : 마스크 (브러시로 칠한 영역)
  3) text   : 텍스트 지시만 ("왼쪽 창가에")
세 방식으로 각각 생성하고 eval/results/ 에 저장한다.

세 프롬프트는 '위치를 알려주는 문장'만 다르고 나머지 문구는 동일하다.
자동 채점은 하지 않는다. eval/results/scoresheet.csv 에 사람이 점수를 적는다.

사용법:
    python eval/run_eval.py              # mock (요금 없음)
    REAL_API=1 python eval/run_eval.py --yes        # 10조합 x 3방식 = 30회 호출
    REAL_API=1 python eval/run_eval.py --all --yes  # 100조합 x 3방식 = 300회 호출
"""

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api_baseline import PROMPT_TEMPLATE, draw_marker, generate, REAL_API  # noqa: E402

RESULTS = ROOT / "eval" / "results"
METHODS = ["marker", "mask", "text"]

# 방마다 고정된 배치 영역 (x%, y%, w%, h%). 실험 재현을 위해 하드코딩한다.
BOXES = [
    (34, 58, 32, 30), (12, 60, 30, 28), (55, 57, 33, 31), (30, 62, 36, 26),
    (60, 60, 28, 30), (18, 58, 34, 32), (42, 61, 30, 28), (25, 59, 32, 30),
    (50, 62, 34, 26), (36, 57, 30, 33),
]

with open(ROOT / "samples" / "items.json", encoding="utf-8") as _f:
    ITEMS = json.load(_f)["items"]          # 단일 출처. app.py도 같은 파일을 읽는다.
ITEM_NAMES = [it["name"] for it in ITEMS]


# --- 방식별 입력 이미지 -----------------------------------------------------

def make_mask_input(room: Image.Image, box) -> Image.Image:
    """브러시로 칠한 것처럼 배치 영역을 불투명 흰색으로 덮는다."""
    x, y, w, h = box
    out = room.convert("RGB").copy()
    ImageDraw.Draw(out).rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
    return out


def where_phrase(room: Image.Image, box) -> str:
    """박스 위치를 텍스트 지시로 바꾼다."""
    cx = (box[0] + box[2] / 2) / room.width
    side = "왼쪽" if cx < 0.38 else ("오른쪽" if cx > 0.62 else "가운데")
    return f"{side} 바닥"


# --- 방식별 프롬프트 (위치 지시 문장만 다르다) --------------------------------

def _with_name(prompt: str, item_name: str) -> str:
    return f"{prompt}\n배치할 가구: {item_name}" if item_name else prompt


def prompt_for(method: str, item_name: str, where: str) -> str:
    if method == "marker":
        return _with_name(PROMPT_TEMPLATE, item_name)
    if method == "mask":
        p = PROMPT_TEMPLATE.replace("반투명 사각형으로 표시된 위치", "흰색으로 칠해진 영역")
        p = p.replace("표시용 사각형은 결과에 남기지 말 것.", "칠해진 흰색 영역은 결과에 남기지 말 것.")
        return _with_name(p, item_name)
    p = PROMPT_TEMPLATE.replace("첫 번째 이미지에서 반투명 사각형으로 표시된 위치", f"첫 번째 이미지의 {where}")
    p = p.replace("\n표시용 사각형은 결과에 남기지 말 것.", "")
    return _with_name(p, item_name)


def pixel_box(room: Image.Image, pct):
    W, H = room.size
    x, y = int(W * pct[0] / 100), int(H * pct[1] / 100)
    return x, y, int(W * pct[2] / 100), int(H * pct[3] / 100)


# --- 비교 그리드 ------------------------------------------------------------

def build_grid(rows):
    """rows: [(pair_id, 원본, marker결과, mask결과, text결과)] -> 비교 이미지 1장."""
    cell_w, pad, head = 384, 8, 28
    labels = ["ROOM", "MARKER", "MASK", "TEXT"]
    thumbs = [[im.copy() for im in row[1:]] for row in rows]
    for row in thumbs:
        for i, im in enumerate(row):
            row[i] = im.resize((cell_w, int(cell_w * im.height / im.width)))
    cell_h = max(im.height for row in thumbs for im in row)

    W = 4 * cell_w + 5 * pad
    H = head + len(rows) * (cell_h + pad) + pad
    grid = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(grid)
    for c, label in enumerate(labels):
        d.text((pad + c * (cell_w + pad) + 6, 8), label, fill=(40, 40, 40))
    for r, row in enumerate(thumbs):
        y = head + r * (cell_h + pad)
        d.text((4, y + 4), rows[r][0][:6], fill=(120, 120, 120))
        for c, im in enumerate(row):
            grid.paste(im, (pad + c * (cell_w + pad), y))
    return grid


# --- 메인 ------------------------------------------------------------------

def _next_run_name(root: Path, prefix: str) -> str:
    """real-001, real-002 … 다음 번호를 찾는다. 기존 결과를 절대 덮지 않는다."""
    n = 1
    while (root / f"{prefix}-{n:03d}").exists():
        n += 1
    return f"{prefix}-{n:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="방 10 x 가구 10 = 100조합 (기본은 10조합)")
    ap.add_argument("--limit", type=int, default=0, help="조합 수 제한")
    ap.add_argument("--yes", action="store_true", help="REAL_API=1 일 때 실제 호출 확인")
    ap.add_argument("--run", default=None,
                    help="결과를 저장할 이름. 생략하면 real-NNN / mock-NNN 이 자동으로 붙는다")
    args = ap.parse_args()


    rooms = sorted((ROOT / "samples" / "rooms").glob("*.png"))[:10]
    items = sorted((ROOT / "samples" / "items").glob("*.png"))[:10]
    if not rooms or not items:
        sys.exit("samples/rooms, samples/items 에 이미지가 없습니다.")

    pairs = list(product(range(len(rooms)), range(len(items)))) if args.all \
        else [(i, i) for i in range(min(len(rooms), len(items)))]
    if args.limit:
        pairs = pairs[: args.limit]

    calls = len(pairs) * len(METHODS)
    if REAL_API and not args.yes:
        sys.exit(f"REAL_API=1 입니다. 실제 호출 {calls}회가 발생합니다. 진행하려면 --yes 를 붙이세요.")
    print(f"조합 {len(pairs)}개 x 방식 {len(METHODS)}개 = {calls}회 "
          f"({'실제 API' if REAL_API else 'mock'})")

    # 폴더는 --yes 확인을 통과한 뒤에 만든다. 중단된 실행이 빈 폴더를 남기면 안 된다.
    # 실행마다 분리 저장한다 — 예전에는 고정 경로에 덮어써서 mock 재실행 한 번에
    # 유료 결과와 손채점 CSV가 함께 사라졌다.
    out_root = RESULTS / (args.run or _next_run_name(RESULTS, "real" if REAL_API else "mock"))
    if out_root.exists() and not args.run:
        sys.exit(f"{out_root} 가 이미 있습니다. --run 으로 다른 이름을 주세요.")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"저장 폴더: {out_root}")

    for m in METHODS:
        (out_root / m).mkdir(parents=True, exist_ok=True)

    grid_rows, sheet = [], []
    for ri, ii in pairs:
        room = Image.open(rooms[ri]).convert("RGB")
        item = Image.open(items[ii]).convert("RGB")
        pair_id = f"{rooms[ri].stem}__{items[ii].stem}"
        box = pixel_box(room, BOXES[ri % len(BOXES)])
        name = ITEM_NAMES[ii % len(ITEM_NAMES)]
        where = where_phrase(room, box)

        outputs = {}
        for method in METHODS:
            if method == "marker":
                src = draw_marker(room, *box)
            elif method == "mask":
                src = make_mask_input(room, box)
            else:
                src = room
            prompt = prompt_for(method, name, where)
            print(f"  [{method}] {pair_id}")
            result = generate(src, item, prompt)
            src.save(out_root / method / f"{pair_id}_input.png")
            result.save(out_root / method / f"{pair_id}_output.png")
            (out_root / method / f"{pair_id}_prompt.txt").write_text(prompt, encoding="utf-8")
            outputs[method] = result
            sheet.append([pair_id, method, name, "", "", ""])

        grid_rows.append((pair_id, room, outputs["marker"], outputs["mask"], outputs["text"]))

    build_grid(grid_rows).save(out_root / "comparison_grid.png")

    with open(out_root / "scoresheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "method", "item", "위치정확도(1-5)", "합성자연스러움(1-5)", "가구보존(1-5)"])
        w.writerows(sheet)

    print(f"\n완료 → {out_root}")
    print("  comparison_grid.png : 방식별 결과 비교")
    print("  scoresheet.csv      : 설문 점수 기입용 (자동 채점 없음)")


if __name__ == "__main__":
    main()
