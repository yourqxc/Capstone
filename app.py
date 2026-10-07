"""RoomFit 로컬 데모. python app.py / Colab: build_app().launch(share=True)."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from roomfit import engine
from roomfit.placement import brush_mask, editable_mask, placement_guide, selection
from roomfit.workflow import run

CSS = """
.gradio-container { width: 100% !important; max-width: 1260px !important; margin: auto; }
#intro { padding: 20px 0 14px; }
#intro h1 { font-size: 38px; letter-spacing: -1.3px; margin-bottom: 8px; }
#intro p { font-size: 16px; color: #6b716c; }
footer { display: none !important; }
"""


def preview(editor):
    try:
        room, box = selection(editor)
        mask = editable_mask(room.size, box)
        image = placement_guide(room, box)
        return image, (
            "빨간 사각형은 가구가 들어갈 위치입니다. 그림자 생성을 위해 주변도 일부 수정합니다. "
            f"사진의 약 {np.mean(np.asarray(mask) > 0) * 100:.0f}%가 편집 영역입니다.")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def synchronize_protection(editor, old_protection):
    """위치 브러시 수정은 유지하되 다른 방을 올리면 보호 마스크를 비웁니다."""
    room = (editor or {}).get("background")
    previous = (old_protection or {}).get("background")
    if room is None:
        return None
    if previous is not None and room.size == previous.size and np.array_equal(np.asarray(room), np.asarray(previous)):
        return gr.skip()
    return {"background": room, "layers": [], "composite": room}


def generate(editor, furniture, description, resolution, seed, protected):
    try:
        room, box = selection(editor)
        if furniture is None:
            raise ValueError("가구 사진을 올려 주세요.")
        if not isinstance(seed, (int, float)) or not math.isfinite(seed) or seed != int(seed):
            raise ValueError("시드는 정수로 입력해 주세요.")
        protect = None
        if protected and any(p is not None for p in protected.get("layers") or []):
            background = protected.get("background")
            if background is None or background.size != room.size or not np.array_equal(
                    np.asarray(background.convert("RGB")), np.asarray(room)):
                raise ValueError("보호 영역이 다른 방 사진에 표시되어 있습니다. 배치 미리보기를 다시 눌러 주세요.")
            protect = brush_mask(protected, room.size)
        result = run(room, furniture, box, description=description or "",
                     longest=int(resolution), seed=int(seed), protection=protect)
        seconds = result["metadata"]["elapsed_seconds"]
        return result["image"], str(result["result"]), str(result["zip"]), (
            f"합성을 완료했습니다 · {seconds:.1f}초\n\n"
            "편집 영역 밖과 보호한 영역은 원본 픽셀을 유지했습니다. "
            "가구의 디자인·위치·접지 그림자가 적절한지 원본과 비교해 주세요.")
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc), duration=None) from exc


def load_example():
    root = Path(__file__).parent / "samples"
    with Image.open(root / "demo-room.png") as image:
        room = image.convert("RGB")
    with Image.open(root / "demo-bed.png") as image:
        furniture = image.convert("RGB")
    # Let users paint on the uploaded canvas; preloaded layers can be offset by
    # ImageEditor's asynchronous background resize.
    return ({"background": room, "layers": [], "composite": None},
            furniture, "gray-brown taupe upholstered bed, neutral gray-brown bedding and headboard")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RoomFit · 가구 배치 시각화", fill_width=True) as demo:
        gr.Markdown("# RoomFit\n방에 놓아 보기\n\n방 사진에 자리를 표시하면 가구의 원근과 조명을 맞춰 새로운 장면을 생성합니다.", elem_id="intro")
        example = gr.Button("제공된 방·침대 예제 불러오기", size="sm")
        with gr.Row():
            with gr.Column(scale=7):
                room = gr.ImageEditor(label="1. 방 사진 · 가구가 차지할 영역을 칠해 주세요", type="pil",
                    sources=["upload", "clipboard"], height=440, transforms=(),
                    brush=gr.Brush(colors=["#ef493e"], default_color="#ef493e", color_mode="fixed"))
            with gr.Column(scale=4):
                furniture = gr.Image(label="2. 가구 사진", type="pil", height=300,
                                     sources=["upload", "clipboard"])
                description = gr.Textbox(label="가구 특징 (선택)", placeholder="예: 짙은 갈색 원목 의자, 베이지색 천 좌판",
                                         max_length=400)
                gr.Markdown("한 가지 가구가 전체로 보이는 사진이 좋습니다. 칠한 영역의 아래쪽을 바닥 접지점에 맞춰 주세요.")
        with gr.Row():
            check = gr.Button("배치 미리보기", variant="secondary")
            create = gr.Button("자연스럽게 합성하기", variant="primary", scale=2)
        status = gr.Markdown("Mac 16GB에서는 512px부터 시작해 주세요. 처음 실행하면 모델을 내려받습니다.")
        with gr.Accordion("배치 확인과 앞쪽 물체 보호", open=False):
            with gr.Row():
                guide = gr.Image(label="선택한 가구 위치", type="pil", interactive=False)
                protected = gr.ImageEditor(label="앞쪽에 그대로 남길 물체만 칠해 주세요 (선택)", type="pil",
                    sources=[], height=360, transforms=(),
                    brush=gr.Brush(colors=["#4c78cf"], default_color="#4c78cf", color_mode="fixed"))
        with gr.Accordion("생성 설정", open=False):
            resolution = gr.Radio([512, 768, 1024], value=512, label="생성 해상도 · 긴 변(px)")
            seed = gr.Number(value=42, precision=0, label="시드 · 같은 설정을 다시 실험할 때 기록하는 값")
            gr.Markdown("Mac과 Colab의 결과가 같아지는 설정은 아닙니다. 1024px는 메모리 사용이 커집니다.")
        with gr.Row():
            output = gr.Image(label="합성 결과", type="pil", interactive=False, height=460)
        with gr.Row():
            download = gr.File(label="결과 PNG", interactive=False)
            evidence = gr.File(label="입력·배치·프롬프트·실행 기록 ZIP", interactive=False)
        gr.Markdown("공개 모델을 이 실행 환경에서 추론합니다. 가구의 세부 무늬와 실제 치수까지 보장하는 설계 도구는 아닙니다.")
        room.change(synchronize_protection, [room, protected], protected)
        example.click(load_example, outputs=[room, furniture, description])
        check.click(preview, room, [guide, status])
        create.click(generate, [room, furniture, description, resolution, seed, protected],
                     [output, download, evidence, status], concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1, max_size=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoomFit 로컬 가구 합성")
    parser.add_argument("--share", action="store_true", help="Colab에서만 필요한 임시 Gradio 공유 주소")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
                       css=CSS, theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone"))
