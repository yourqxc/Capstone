"""AI 셀프 인테리어 시각화 데모.

방 사진 + 가구 사진 + 배치 위치 -> 그 자리에 가구가 놓인 합성 이미지.
REAL_API=1 일 때만 실제 편집 모델 API를 호출한다. 기본은 mock.
"""

import base64
import io
import json
import os
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
REAL_API = os.getenv("REAL_API", "0") == "1"
MODEL = os.getenv("MODEL", "gemini-3.1-flash-image")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"

ROOT = Path(__file__).parent
SAMPLE_ROOMS = ROOT / "samples" / "rooms"
SAMPLE_ITEMS = ROOT / "samples" / "items"

PROMPT_TEMPLATE = """두 번째 이미지의 가구를 첫 번째 이미지에서 반투명 사각형으로 표시된 위치에 배치해줘.
가구의 형태, 색상, 재질은 원본 그대로 유지할 것.
방의 원근, 조명 방향, 바닥 접촉 그림자를 맞춰 자연스럽게 합성할 것.
방의 구조(벽, 창문, 바닥)와 나머지 부분은 변경하지 말 것.
표시용 사각형은 결과에 남기지 말 것."""


# --- 핵심 함수 3개 ---------------------------------------------------------

def draw_marker(room: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """방 사진 위에 반투명 사각형을 그린다. 모델에게 배치 위치를 알려주는 신호."""
    base = room.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    line = max(2, base.width // 200)
    ImageDraw.Draw(overlay).rectangle(
        [x, y, x + w, y + h],
        fill=(255, 80, 80, 90),
        outline=(255, 40, 40, 255),
        width=line,
    )
    return Image.alpha_composite(base, overlay).convert("RGB")


def build_prompt(item_name: str = "") -> str:
    """편집 모델에 보낼 지시문 생성."""
    item_name = (item_name or "").strip()
    if item_name:
        return f"{PROMPT_TEMPLATE}\n배치할 가구: {item_name}"
    return PROMPT_TEMPLATE


def generate(room_marked: Image.Image, item: Image.Image, prompt: str) -> Image.Image:
    """편집 모델 API 호출. REAL_API != 1 이면 room_marked를 그대로 반환."""
    if not REAL_API:
        print("[mock] REAL_API=1 이 아니라서 API를 호출하지 않았습니다.")
        return room_marked
    if not API_KEY:
        raise gr.Error("API_KEY가 .env에 없습니다.")

    body = {
        "model": MODEL,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": "image/png", "data": _to_b64(room_marked)},
            {"type": "image", "mime_type": "image/png", "data": _to_b64(item)},
        ],
    }
    res = requests.post(
        ENDPOINT,
        headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=180,
    )
    if res.status_code != 200:
        print(f"[API {res.status_code}] {res.text[:4000]}")
        raise gr.Error(f"API 오류 {res.status_code}. 터미널 로그를 확인하세요.")

    data = res.json()
    b64 = _find_image(data)
    if not b64:
        print("[API 응답 전문]")
        print(json.dumps(data, ensure_ascii=False)[:4000])
        raise gr.Error("응답에서 이미지를 찾지 못했습니다. 터미널 로그를 확인하세요.")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _find_image(node):
    """응답 JSON의 어느 위치에 있든 base64 이미지 조각을 찾는다."""
    if isinstance(node, dict):
        if node.get("type") == "image" and isinstance(node.get("data"), str):
            return node["data"]
        if isinstance(node.get("inline_data"), dict):
            return node["inline_data"].get("data")
        for value in node.values():
            found = _find_image(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_image(value)
            if found:
                return found
    return None


# --- 위치 지정 -------------------------------------------------------------

def brush_box(editor_value):
    """ImageEditor에서 브러시로 칠한 영역의 사각형 범위. 안 칠했으면 None."""
    box = None
    for layer in (editor_value or {}).get("layers") or []:
        if layer is None:
            continue
        found = layer.convert("RGBA").getchannel("A").getbbox()
        if found is None:
            continue
        box = found if box is None else (
            min(box[0], found[0]), min(box[1], found[1]),
            max(box[2], found[2]), max(box[3], found[3]),
        )
    return box


def percent_box(room: Image.Image, x_pct, y_pct, w_pct, h_pct):
    """슬라이더 백분율을 픽셀 좌표로."""
    W, H = room.size
    x, y = int(W * x_pct / 100), int(H * y_pct / 100)
    w, h = int(W * w_pct / 100), int(H * h_pct / 100)
    return x, y, min(w, W - x - 1), min(h, H - y - 1)


# --- Gradio 이벤트 핸들러 ---------------------------------------------------

def run(editor_value, item, item_name, x_pct, y_pct, w_pct, h_pct):
    if not editor_value or editor_value.get("background") is None:
        raise gr.Error("방 사진을 올려주세요.")
    if item is None:
        raise gr.Error("가구 사진을 올려주세요.")

    room = editor_value["background"].convert("RGB")
    painted = brush_box(editor_value)
    if painted:
        x, y, w, h = painted[0], painted[1], painted[2] - painted[0], painted[3] - painted[1]
        how = "브러시로 칠한 영역"
    else:
        x, y, w, h = percent_box(room, x_pct, y_pct, w_pct, h_pct)
        how = "슬라이더 좌표"

    marked = draw_marker(room, x, y, w, h)
    prompt = build_prompt(item_name)
    result = generate(marked, item, prompt)

    mode = "실제 API" if REAL_API else "mock (REAL_API=0 이라 원본을 그대로 반환)"
    log = f"위치: {how} → x={x}, y={y}, w={w}, h={h}\n모드: {mode}\n\n{prompt}"
    return room, marked, result, log


def _examples():
    rooms = sorted(SAMPLE_ROOMS.glob("*.png"))
    items = sorted(SAMPLE_ITEMS.glob("*.png"))
    names = ["소파", "1인용 의자", "원목 테이블"]
    return [
        [str(r), str(i), n]
        for r, i, n in zip(rooms, items, names)
    ]


with gr.Blocks(title="AI 셀프 인테리어 시각화") as demo:
    gr.Markdown(
        "# AI 셀프 인테리어 시각화\n"
        "방 사진과 가구 사진을 올리고, 가구를 놓을 자리를 브러시로 칠하거나 슬라이더로 지정하세요."
    )

    if not REAL_API:
        gr.Markdown(
            "### 지금은 mock 모드입니다\n"
            "API를 호출하지 않고, 배치 위치가 표시된 방 사진을 그대로 돌려줍니다. "
            "실제로 합성하려면 `.env`에 `API_KEY`를 넣고 `REAL_API=1`로 바꾼 뒤 다시 실행하세요."
        )

    with gr.Row():
        with gr.Column():
            room_editor = gr.ImageEditor(
                label="방 사진 (놓을 자리를 브러시로 칠하세요)",
                type="pil",
                layers=False,
                brush=gr.Brush(colors=["#ff4d4d"], default_size=40),
            )
            gr.Markdown("브러시를 쓰지 않으면 아래 슬라이더 좌표를 사용합니다.")
            x_pct = gr.Slider(0, 95, value=35, step=1, label="가로 위치 %")
            y_pct = gr.Slider(0, 95, value=55, step=1, label="세로 위치 %")
            w_pct = gr.Slider(5, 100, value=30, step=1, label="너비 %")
            h_pct = gr.Slider(5, 100, value=30, step=1, label="높이 %")
        with gr.Column():
            item_image = gr.Image(label="가구 / 소품 사진", type="pil")
            item_name = gr.Textbox(label="가구 이름 (선택)", placeholder="예: 그린 벨벳 소파", lines=1, max_lines=1)
            run_btn = gr.Button("가구 배치 생성", variant="primary")

    with gr.Row():
        before = gr.Image(label="Before", type="pil")
        marker_view = gr.Image(label="배치 위치 표시", type="pil")
        after = gr.Image(label="After", type="pil")

    log_box = gr.Textbox(label="사용된 좌표 / 프롬프트", lines=8)

    run_btn.click(
        run,
        inputs=[room_editor, item_image, item_name, x_pct, y_pct, w_pct, h_pct],
        outputs=[before, marker_view, after, log_box],
    )

    examples = _examples()
    if examples:
        gr.Examples(examples=examples, inputs=[room_editor, item_image, item_name], label="샘플")


if __name__ == "__main__":
    print(f"REAL_API={'1 (실제 호출)' if REAL_API else '0 (mock)'}  MODEL={MODEL}")
    demo.launch()
