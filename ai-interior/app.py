"""AI 셀프 인테리어 시각화 데모.

방 사진 + 가구 사진 + 배치 위치 -> 그 자리에 가구가 놓인 합성 이미지.

두 가지 방식을 고를 수 있다.
  로컬 파이프라인 : Depth Anything -> 바닥 평면(OpenCV RANSAC) -> SAM 누끼 -> 원근 배치 -> 합성
  Gemini API      : 마커를 그린 방 사진과 가구 사진을 편집 모델에 넘긴다 (비교 대상)
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

import api_baseline as api
from pipeline.compose import compose
from pipeline.depth import estimate_depth
from pipeline.geometry import auto_height_px, floor_plane, place_transform
from pipeline.segment import (background_risk, clamp_box, cutout, default_box,
                              has_alpha, segment)

ROOT = Path(__file__).parent
SAMPLE_ROOMS = ROOT / "samples" / "rooms"
SAMPLE_ITEMS = ROOT / "samples" / "items"


def load_items():
    """가구 목록의 단일 출처. samples/items.json 하나만 고치면 앱과 실험이 함께 따라온다.

    이전에는 app.py와 eval/run_eval.py가 각자 이름 목록을 들고 있었고 10개가 전부 어긋나
    있었다. 이 이름은 API 프롬프트의 "배치할 가구: OO"에 그대로 들어가므로,
    어긋나면 사진과 지시문이 불일치한 조건에서 실험이 돌아간다.
    """
    with open(ROOT / "samples" / "items.json", encoding="utf-8") as f:
        return json.load(f)["items"]


ITEMS = load_items()
ITEM_NAMES = [it["name"] for it in ITEMS]


# --- 입력 해석 -------------------------------------------------------------

def brush_box(editor_value):
    """ImageEditor에서 브러시로 칠한 영역의 사각형 범위. 안 칠했으면 None."""
    box = None
    for layer in (editor_value or {}).get("layers") or []:
        if layer is None:
            continue
        found = layer.convert("RGBA").getchannel("A").getbbox()
        if found is None:
            continue
        box = found if box is None else (min(box[0], found[0]), min(box[1], found[1]),
                                         max(box[2], found[2]), max(box[3], found[3]))
    return box


def percent_box(room, x_pct, y_pct, w_pct, h_pct):
    """슬라이더 백분율을 픽셀 좌표로 (x0, y0, x1, y1)."""
    W, H = room.size
    x, y = int(W * x_pct / 100), int(H * y_pct / 100)
    return x, y, min(x + int(W * w_pct / 100), W - 1), min(y + int(H * h_pct / 100), H - 1)


def _background(editor_value, keep_alpha: bool = False):
    """ImageEditor의 배경 이미지. keep_alpha=True면 투명도를 살린다 (누끼된 PNG용)."""
    if not editor_value or editor_value.get("background") is None:
        return None
    img = editor_value["background"]
    return img.convert("RGBA") if keep_alpha else img.convert("RGB")


def preview(room, box, plane=None):
    """배치 위치와 추정된 바닥을 눈으로 확인하는 그림."""
    arr = np.array(room).astype(np.int16)
    if plane is not None and plane.get("mask") is not None:
        m = plane["mask"]
        arr[m, 1] = np.minimum(255, arr[m, 1] + 70)
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline=(255, 40, 40), width=max(2, room.size[0] // 200))
    if plane is not None and plane.get("horizon_y") is not None:
        hy = plane["horizon_y"]
        if 0 <= hy < room.size[1]:
            d.line([(0, hy), (room.size[0], hy)], fill=(255, 140, 0), width=2)
    return img


# --- 이벤트 핸들러 ---------------------------------------------------------

def check_placement(alpha, room_size, log):
    """배치 결과가 화면 밖으로 잘리거나 지나치게 작으면 알려준다."""
    ys, xs = np.where(alpha > 0.5)
    if len(ys) == 0:
        log.append("경고: 가구가 화면 밖에 배치됐습니다. 박스를 안쪽으로 옮기세요.")
        return
    W, H = room_size
    clipped = [n for n, c in
               (("왼쪽", xs.min() <= 1), ("오른쪽", xs.max() >= W - 2),
                ("위", ys.min() <= 1), ("아래", ys.max() >= H - 2)) if c]
    if clipped:
        log.append(f"경고: 가구가 {'/'.join(clipped)} 화면 끝에서 잘렸습니다.")
    frac = (ys.max() - ys.min()) / H
    if frac < 0.18:
        log.append(f"경고: 가구 높이가 화면의 {frac*100:.0f}%뿐입니다. "
                   "박스를 가구 크기만큼 크게 칠하거나 크기 배율을 올리세요.")


def run(room_ed, item_ed, item_name, engine, mode, candidate, size_mode, size_scale, pay_ok,
        x_pct, y_pct, w_pct, h_pct, progress=gr.Progress()):
    room = _background(room_ed)
    item = _background(item_ed, keep_alpha=True)
    if room is None:
        raise gr.Error("방 사진을 올려주세요.")
    if item is None:
        raise gr.Error("가구 사진을 올려주세요.")

    painted = brush_box(room_ed)
    if painted:
        box, how = painted, "브러시로 칠한 영역"
    else:
        box, how = percent_box(room, x_pct, y_pct, w_pct, h_pct), "슬라이더 좌표"

    item_box = brush_box(item_ed)
    log = [f"배치 위치: {how} → {box}"]

    if engine.startswith("로컬"):
        progress(0.1, desc="깊이 추정")
        depth = estimate_depth(room)
        progress(0.4, desc="바닥 평면 추정")
        plane = floor_plane(room, depth)
        if plane["coef"] is None:
            raise gr.Error("바닥 평면을 찾지 못했습니다. 바닥이 더 보이는 사진을 써 주세요.")

        progress(0.6, desc="가구 분리 (SAM)")
        cand = None if candidate == "자동" else int(candidate)
        raw_box = item_box or default_box(item.size)
        used_box = clamp_box(raw_box, item.size)
        if has_alpha(item):
            risk = 0.0
            log.append("가구 사진에 이미 투명 배경이 있어 SAM을 건너뛰고 그 알파를 씁니다.")
        else:
            risk = background_risk(segment(item, raw_box, cand), used_box)
        if risk > 0.55:
            log.append(f"경고: 가구 대신 배경이 잡혔을 수 있습니다 (배경 위험도 {risk:.2f}). "
                       "SAM 후보를 0/1/2로 바꾸거나 가구에 더 딱 맞게 칠해보세요.")
        rgba = cutout(item, box=raw_box, candidate=cand, crop=True)

        progress(0.85, desc="원근 배치 · 합성")
        # 자동 모드면 크기를 바닥 평면과 지평선에서 계산한다(기획서의 "크기 자동 보정").
        # 가구의 실제 높이는 samples/items.json 의 height_m 에서 온다.
        height_px = None
        if size_mode.startswith("자동"):
            h_m = next((it["height_m"] for it in ITEMS if it["name"] == (item_name or "").strip()),
                       None)
            if h_m:
                height_px = auto_height_px(plane, box[3], room.size, h_m)
                if height_px is None:
                    log.append("경고: 지평선 기준 자동 크기를 계산할 수 없어 박스 크기를 씁니다.")
            else:
                log.append(f"경고: '{item_name}'의 실제 높이를 모릅니다. 박스 크기를 씁니다. "
                           "(가구 이름을 샘플과 같게 적으면 자동 크기가 켜집니다)")
        M = place_transform(plane, box, rgba.size, room.size,
                            mode="flat" if mode.startswith("바닥") else "upright",
                            scale=size_scale, height_px=height_px)
        result = compose(room, rgba, M, plane, depth=depth)
        from pipeline.compose import _warp_rgba
        check_placement(_warp_rgba(rgba, M, room.size)[1], room.size, log)

        log += [
            f"가구 박스: {'브러시' if item_box else '기본(중앙 80%)'} → {used_box}"
            + ("  (가장자리에서 8% 안쪽으로 조정됨)" if used_box != raw_box else ""),
            f"배경 위험도: {risk:.2f} (0.55 넘으면 배경일 수 있음)",
            f"SAM 후보: {candidate}",
            f"지평선 y: {plane['horizon_y']:.0f}" if plane["horizon_y"] else "지평선: 계산 불가",
            f"바닥 추정 면적: {plane['mask'].mean() * 100:.1f}%",
            f"배치 방식: {mode}   크기 배율: {size_scale:.2f}",
            (f"크기: 자동 (실제 높이 기준, 화면 {height_px:.0f}px)" if height_px
             else "크기: 박스 기준 (수동)"),
        ]
        return room, preview(room, box, plane), result, "\n".join(log)

    # --- Gemini API 경로 (비교 대상) ---
    if api.REAL_API and not pay_ok:
        raise gr.Error(f"유료 호출입니다 (약 ${api.COST_PER_CALL:.3f}). "
                       "'유료 호출에 동의합니다'를 켜거나 로컬 파이프라인을 쓰세요.")
    x0, y0, x1, y1 = box
    marked = api.draw_marker(room, box)
    item = item.convert("RGB")
    prompt = api.build_prompt(item_name)
    progress(0.5, desc="API 호출")
    try:
        result = api.generate(marked, item, prompt)
    except RuntimeError as e:
        raise gr.Error(str(e))
    log += [f"모드: {'실제 API' if api.REAL_API else 'mock (REAL_API=0 이라 원본 반환)'}",
            f"모델: {api.MODEL}",
            f"이번 세션 유료 호출 {api.call_count()}회 · 누적 약 ${api.spent():.2f}",
            "", prompt]
    return room, marked, result, "\n".join(log)


def _examples():
    rooms = sorted(SAMPLE_ROOMS.glob("*.png"))
    items = sorted(SAMPLE_ITEMS.glob("*.png"))
    return [[str(r), str(i), n] for r, i, n in zip(rooms, items, ITEM_NAMES)]


with gr.Blocks(title="AI 셀프 인테리어 시각화") as demo:
    gr.Markdown(
        "# AI 셀프 인테리어 시각화\n"
        "방 사진에는 **가구가 차지할 크기만큼** 칠하세요 (작게 칠하면 작은 가구가 놓입니다). "
        "가구 사진에는 **가구를 감싸는 영역**을 칠하세요."
    )
    if api.REAL_API:
        gr.Markdown(
            f"### ⚠️ 유료 모드입니다 — Gemini 경로는 호출 1회당 약 ${api.COST_PER_CALL:.3f}\n"
            "아래 **확인란을 켜야** Gemini 경로가 실행됩니다. 로컬 파이프라인은 무료라 제한이 없습니다."
        )
    else:
        gr.Markdown("Gemini API는 현재 mock입니다 (`.env`의 `REAL_API=1`로 실제 호출). "
                    "로컬 파이프라인은 mock 없이 항상 실제로 동작합니다.")

    with gr.Row():
        with gr.Column():
            room_ed = gr.ImageEditor(label="방 사진 — 놓을 자리를 칠하세요", type="pil",
                                     layers=False, brush=gr.Brush(colors=["#ff4d4d"], default_size=40))
            gr.Markdown("칠하지 않으면 아래 슬라이더를 씁니다.")
            with gr.Row():
                x_pct = gr.Slider(0, 95, value=35, step=1, label="가로 %")
                y_pct = gr.Slider(0, 95, value=55, step=1, label="세로 %")
            with gr.Row():
                w_pct = gr.Slider(5, 100, value=30, step=1, label="너비 %")
                h_pct = gr.Slider(5, 100, value=30, step=1, label="높이 %")
        with gr.Column():
            item_ed = gr.ImageEditor(label="가구 사진 — 가구를 감싸게 칠하세요", type="pil",
                                     layers=False, brush=gr.Brush(colors=["#4d9dff"], default_size=60))
            item_name = gr.Textbox(label="가구 이름 (API 경로에서만 사용)",
                                   placeholder="예: 원목 의자", lines=1, max_lines=1)
            engine = gr.Radio(["로컬 파이프라인", "Gemini API (비교)"],
                              value="로컬 파이프라인", label="합성 방식")
            pay_ok = gr.Checkbox(value=False, visible=api.REAL_API,
                                 label=f"유료 호출에 동의합니다 (1회 약 ${api.COST_PER_CALL:.3f})")
            with gr.Row():
                mode = gr.Radio(["세워놓기 (의자·책장)", "바닥에 깔기 (러그)"],
                                value="세워놓기 (의자·책장)", label="배치 방식")
                candidate = gr.Radio(["자동", "0", "1", "2"], value="자동",
                                     label="SAM 후보 (누끼가 이상하면 바꿔보세요)")
            size_mode = gr.Radio(["자동 (바닥 평면 기준)", "수동 (칠한 박스 크기)"],
                                 value="자동 (바닥 평면 기준)", label="크기 결정 방식")
            size_scale = gr.Slider(0.4, 2.5, value=1.0, step=0.05,
                                   label="크기 미세조정 배율")
            run_btn = gr.Button("가구 배치 생성", variant="primary")

    with gr.Row():
        before = gr.Image(label="Before", type="pil")
        middle = gr.Image(label="배치 위치 · 추정 바닥(초록) · 지평선(주황)", type="pil")
        after = gr.Image(label="After", type="pil")

    log_box = gr.Textbox(label="처리 내역", lines=9)

    run_btn.click(run,
                  inputs=[room_ed, item_ed, item_name, engine, mode, candidate,
                          size_mode, size_scale, pay_ok, x_pct, y_pct, w_pct, h_pct],
                  outputs=[before, middle, after, log_box])

    if (ex := _examples()):
        gr.Examples(examples=ex, inputs=[room_ed, item_ed, item_name], label="샘플")


if __name__ == "__main__":
    print(f"Gemini API REAL_API={'1' if api.REAL_API else '0'}  MODEL={api.MODEL}")
    print("로컬 파이프라인 첫 실행 시 모델 로드에 20초 정도 걸립니다.")
    demo.launch()
