"""사람 평가 설문 (DEVLOG §31). 로컬에서 띄우고 링크를 나눠 준다. 응답은 이 컴퓨터에만 쌓인다.

    python survey.py            # http://127.0.0.1:7861  (같은 와이파이의 다른 기기는 --host 0.0.0.0)
    python survey.py --share    # 외부에서 접속 가능한 임시 공개 링크 (gradio 터널)

문항과 이미지는 eval/survey/ (make_stimuli.py가 만든다). 화면에는 어느 방식인지 표시하지 않고,
참여자마다 순서를 섞는다. 이름 같은 개인정보는 받지 않는다 — 임의의 참여자 번호만 남긴다.
응답: eval/survey/responses.csv  (문항 하나 답할 때마다 한 줄씩 바로 저장 — 중간에 나가도 남는다)
집계: python eval/survey/summarize.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

HERE = Path(__file__).parent / "eval" / "survey"
TRIALS = json.loads((HERE / "trials.json").read_text(encoding="utf-8"))
STIM = HERE / "stimuli"
RESP = HERE / "responses.csv"
FIELDS = ["time", "rater", "order", "trial", "q_position", "q_natural", "q_identity", "seconds"]
SCALE = ["1 전혀 아니다", "2", "3 보통", "4", "5 매우 그렇다"]
_lock = threading.Lock()

INTRO = """## 가구 합성 결과 평가
방 사진에 가구를 합성한 결과를 보고 세 질문에 1~5점으로 답해 주세요. 정답은 없습니다.
- **방 사진의 빨간 표시**는 가구를 놓아 달라고 요청한 자리입니다.
- **가구 사진**은 놓아야 할 가구입니다.
- 모두 %d장이고 5~8분 걸립니다. 이름 등 개인정보는 받지 않습니다.""" % len(TRIALS)


def _save(row: dict):
    with _lock:
        new = not RESP.exists()
        with open(RESP, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def _view(st):
    t = TRIALS[st["order"][st["i"]]]
    return (f"### {st['i'] + 1} / {len(TRIALS)}",
            str(STIM / t["room"]), str(STIM / t["item_img"]), str(STIM / t["result"]))


def start(st):
    rater = uuid.uuid4().hex[:8]
    order = list(range(len(TRIALS)))
    random.Random(rater).shuffle(order)
    st = {"rater": rater, "order": order, "i": 0, "t0": time.time()}
    head, room, item, res = _view(st)
    return (st, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            head, room, item, res, None, None, None)


def nxt(st, q1, q2, q3):
    # gr.Error를 던지면 gradio가 출력 칸(질문 포함)을 모두 오류 상태로 바꿔 더 답할 수 없게 된다.
    # 알림만 띄우고 화면은 그대로 둔다.
    if not st or not (q1 and q2 and q3):
        gr.Warning("세 질문에 모두 답해 주세요.")
        keep = gr.update()
        return st, keep, keep, keep, keep, keep, keep, q1, q2, q3
    t = TRIALS[st["order"][st["i"]]]
    _save({"time": datetime.now().isoformat(timespec="seconds"), "rater": st["rater"], "order": st["i"],
           "trial": t["id"], "q_position": q1[0], "q_natural": q2[0], "q_identity": q3[0],
           "seconds": round(time.time() - st["t0"], 1)})
    st = {**st, "i": st["i"] + 1, "t0": time.time()}
    if st["i"] >= len(TRIALS):
        return (st, gr.update(visible=False), gr.update(visible=True),
                "", None, None, None, None, None, None)
    head, room, item, res = _view(st)
    return (st, gr.update(visible=True), gr.update(visible=False), head, room, item, res, None, None, None)


with gr.Blocks(title="가구 합성 평가") as demo:
    st = gr.State(None)
    with gr.Column(visible=True) as intro:
        gr.Markdown(INTRO)
        go = gr.Button("시작", variant="primary")
    with gr.Column(visible=False) as body:
        head = gr.Markdown()
        with gr.Row():
            room = gr.Image(label="방 사진 (빨간 표시 = 놓을 자리)", interactive=False, height=220)
            item = gr.Image(label="가구 사진", interactive=False, height=220)
        res = gr.Image(label="합성 결과", interactive=False, height=460)
        q1 = gr.Radio(SCALE, label="1. 가구가 빨간 표시 자리에 놓였나요?")
        q2 = gr.Radio(SCALE, label="2. 그 방에 실제로 있는 가구처럼 자연스러운가요? (크기, 바닥에 닿은 모습, 밝기)")
        q3 = gr.Radio(SCALE, label="3. 가구 사진 속 가구와 같은 제품으로 보이나요?")
        nb = gr.Button("다음", variant="primary")
    with gr.Column(visible=False) as done:
        gr.Markdown("## 감사합니다\n응답이 저장됐습니다. 창을 닫으셔도 됩니다.")

    go.click(start, st, [st, intro, body, done, head, room, item, res, q1, q2, q3])
    nb.click(nxt, [st, q1, q2, q3], [st, body, done, head, room, item, res, q1, q2, q3])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="gradio 임시 공개 링크")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    demo.launch(server_name=a.host, server_port=a.port, share=a.share, allowed_paths=[str(STIM)])
