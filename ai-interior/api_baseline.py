"""Gemini 이미지 편집 API 경로 — 로컬 파이프라인의 비교 대상(baseline).

이 파일은 Gradio에 의존하지 않는다. eval에서 UI 없이 불러 쓰기 위해서다.
실패는 RuntimeError로 던지고, UI로 바꾸는 것은 app.py가 한다.

요금이 발생한다. 기본은 mock이며 REAL_API=1 일 때만 실제로 호출한다.
이미지 모델은 무료 등급 할당량이 0이므로 결제 등록이 필요하다 (DEVLOG §2-c).
"""

from __future__ import annotations

import base64
import io
import json
import os

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
REAL_API = os.getenv("REAL_API", "0") == "1"
MODEL = os.getenv("MODEL", "gemini-3.1-flash-image")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"

# 장당 요금 (2026-09 기준). gemini-3.1-flash-image $0.067 / gemini-2.5-flash-image $0.039
COST_PER_CALL = 0.039 if "2.5" in MODEL else 0.067

_CALLS = 0


def call_count() -> int:
    """이번 프로세스에서 실제로 발생한 유료 호출 횟수."""
    return _CALLS


def spent() -> float:
    """이번 프로세스 누적 예상 비용(달러)."""
    return _CALLS * COST_PER_CALL

PROMPT_TEMPLATE = """두 번째 이미지의 가구를 첫 번째 이미지에서 반투명 사각형으로 표시된 위치에 배치해줘.
가구의 형태, 색상, 재질은 원본 그대로 유지할 것.
방의 원근, 조명 방향, 바닥 접촉 그림자를 맞춰 자연스럽게 합성할 것.
방의 구조(벽, 창문, 바닥)와 나머지 부분은 변경하지 말 것.
표시용 사각형은 결과에 남기지 말 것."""


def draw_marker(room: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """방 사진 위에 반투명 사각형을 그린다. 모델에게 배치 위치를 알려주는 신호."""
    base = room.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [x, y, x + w, y + h], fill=(255, 80, 80, 90),
        outline=(255, 40, 40, 255), width=max(2, base.width // 200))
    return Image.alpha_composite(base, overlay).convert("RGB")


def build_prompt(item_name: str = "") -> str:
    """편집 모델에 보낼 지시문 생성."""
    item_name = (item_name or "").strip()
    return f"{PROMPT_TEMPLATE}\n배치할 가구: {item_name}" if item_name else PROMPT_TEMPLATE


def generate(room_marked: Image.Image, item: Image.Image, prompt: str) -> Image.Image:
    """편집 모델 API 호출. REAL_API != 1 이면 room_marked를 그대로 반환."""
    if not REAL_API:
        print("[mock] REAL_API=1 이 아니라서 API를 호출하지 않았습니다.")
        return room_marked
    if not API_KEY:
        raise RuntimeError("API_KEY가 .env에 없습니다.")

    body = {"model": MODEL, "input": [
        {"type": "text", "text": prompt},
        {"type": "image", "mime_type": "image/png", "data": _to_b64(room_marked)},
        {"type": "image", "mime_type": "image/png", "data": _to_b64(item)},
    ]}
    global _CALLS
    _CALLS += 1
    res = requests.post(ENDPOINT, headers={"x-goog-api-key": API_KEY,
                                           "Content-Type": "application/json"},
                        json=body, timeout=180)
    if res.status_code != 200:
        print(f"[API {res.status_code}] {res.text[:4000]}")
        raise RuntimeError(f"API 오류 {res.status_code}. 터미널 로그를 확인하세요.")

    data = res.json()
    b64 = _find_image(data)
    if not b64:
        print("[API 응답 전문]")
        print(json.dumps(data, ensure_ascii=False)[:4000])
        raise RuntimeError("응답에서 이미지를 찾지 못했습니다. 터미널 로그를 확인하세요.")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _find_image(node):
    """응답 JSON의 어느 위치에 있든 base64 이미지 조각을 찾는다.

    실제 200 응답 구조는 steps[].content[]{type:"image", data} 이며,
    입력 이미지는 응답에 포함되지 않는 것을 확인했다 (DEVLOG §2-d).
    """
    if isinstance(node, dict):
        if node.get("type") == "image" and isinstance(node.get("data"), str):
            return node["data"]
        if isinstance(node.get("inline_data"), dict):
            return node["inline_data"].get("data")
        for value in node.values():
            if (found := _find_image(value)):
                return found
    elif isinstance(node, list):
        for value in node:
            if (found := _find_image(value)):
                return found
    return None
