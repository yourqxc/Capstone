"""사람 평가용 자극 이미지 만들기 (DEVLOG §31).

CLIP은 사실감과 가구 보존을 재지 못한다(§23). 비교의 근거는 사람 채점이다.
결과를 보기 전에 정한 구성:

  A. 방식 비교 — 유료 파일럿 3쌍 x {로컬 가구 박스, 로컬 + 다듬기 0.35, Gemini 마커} = 9
  B. 다듬기 효과 — 누끼 정상 5쌍(04 05 06 08 09) x {다듬기 없음, 있음} = 10

로컬 결과는 지금 파이프라인(크기 기준 §30 포함)으로 새로 만든다. Gemini는 파일럿에서
받아 둔 이미지를 쓴다(다시 부르지 않음). 파일럿 3쌍은 Gemini가 받은 배치 박스를 쓴다.

    python eval/survey/make_stimuli.py     # diffusers와 SD 모델 필요 (약 3분)

결과: eval/survey/stimuli/*.jpg 와 trials.json (문항 id -> 조건. 설문 화면에는 조건을 보이지 않는다)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hashlib

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api_baseline import draw_marker  # noqa: E402
from eval.rescore_pilot import PILOT, PILOT_BOXES, local_now  # noqa: E402
from eval.run_eval import BOXES, ITEMS, pixel_box  # noqa: E402

OUT = Path(__file__).parent / "stimuli"
MAXW = 900


def blind(tid: str) -> str:
    """결과 이미지 파일 이름. 화면의 이미지 주소로 조건(A01g = Gemini 등)이 드러나지 않게 한다."""
    return "r_" + hashlib.sha1(("stim-" + tid).encode()).hexdigest()[:10] + ".jpg"


def save(im: Image.Image, name: str) -> str:
    im = im.convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, int(im.height * MAXW / im.width)), Image.LANCZOS)
    im.save(OUT / name, quality=85)
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trials = []

    def pair_refs(i, box):
        room = Image.open(ROOT / f"samples/rooms/room_{i:02d}.png").convert("RGB")
        item = Image.open(ROOT / f"samples/items/item_{i:02d}.png").convert("RGB")
        return room, item, save(draw_marker(room, box), f"p{i:02d}_room.jpg"), save(item, f"p{i:02d}_item.jpg")

    # A. 방식 비교
    for i, pct in PILOT_BOXES.items():
        meta = ITEMS[i - 1]
        room = Image.open(ROOT / f"samples/rooms/room_{i:02d}.png").convert("RGB")
        box = pixel_box(room, pct)
        room, item, room_f, item_f = pair_refs(i, box)
        outs = {
            "local": local_now(room, item, meta, box, painted=True),
            "local_refine": local_now(room, item, meta, box, painted=True, refine_strength=0.35),
            "gemini": Image.open(PILOT / "api_marker" / f"room_{i:02d}__item_{i:02d}_output.png").resize(room.size),
        }
        for cond, im in outs.items():
            tid = f"A{i:02d}{cond[0]}{'r' if cond == 'local_refine' else ''}"
            trials.append({"id": tid, "part": "A", "pair": i, "condition": cond, "item": meta["name"],
                           "room": room_f, "item_img": item_f, "result": save(im, blind(tid))})
        print(f"A pair {i:02d} 완료", flush=True)

    # B. 다듬기 효과
    for i in (4, 5, 6, 8, 9):
        meta = ITEMS[i - 1]
        room = Image.open(ROOT / f"samples/rooms/room_{i:02d}.png").convert("RGB")
        box = pixel_box(room, BOXES[i - 1])
        room, item, room_f, item_f = pair_refs(i, box)
        for cond, st in (("local", None), ("local_refine", 0.35)):
            tid = f"B{i:02d}{'r' if st else 'l'}"
            im = local_now(room, item, meta, box, painted=True, refine_strength=st)
            trials.append({"id": tid, "part": "B", "pair": i, "condition": cond, "item": meta["name"],
                           "room": room_f, "item_img": item_f, "result": save(im, blind(tid))})
        print(f"B pair {i:02d} 완료", flush=True)

    (Path(__file__).parent / "trials.json").write_text(json.dumps(trials, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    print(f"문항 {len(trials)}개 -> {OUT}")


if __name__ == "__main__":
    main()
