"""이미지 좌표, 배치 가이드, 생성 영역 밖 원본 보존. 모델에 의존하지 않습니다."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps


@dataclass(frozen=True)
class Box:
    """이미지에 대한 정규화 좌표 (왼쪽, 위, 오른쪽, 아래), 범위 0~1."""
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        values = (self.left, self.top, self.right, self.bottom)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("배치 좌표는 유한한 숫자여야 합니다.")
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValueError("배치 영역은 사진 안에 있어야 하며 폭과 높이가 필요합니다.")

    def pixels(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        w, h = size
        return (int(self.left * w), int(self.top * h),
                min(w, math.ceil(self.right * w)), min(h, math.ceil(self.bottom * h)))

    def as_list(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


def clean_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise ValueError("사진을 먼저 올려 주세요.")
    if min(image.size) < 64 or image.width * image.height > 40_000_000:
        raise ValueError("사진은 각 변이 64px 이상, 전체 4천만 픽셀 이하여야 합니다.")
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA" or "transparency" in image.info:
        white = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(white, image.convert("RGBA"))
    return image.convert("RGB")


def brush_mask(editor: dict | None, size: tuple[int, int]) -> Image.Image:
    """Gradio 브러시 레이어의 알파만 읽습니다. 원본/composite 색상은 읽지 않습니다."""
    mask = Image.new("L", size, 0)
    for layer in (editor or {}).get("layers") or []:
        if layer is None:
            continue
        if not isinstance(layer, Image.Image) or layer.size != size:
            raise ValueError("선택 영역과 방 사진의 크기가 다릅니다. 다시 표시해 주세요.")
        alpha = layer.convert("RGBA").getchannel("A")
        mask = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(alpha)))
    return mask


def selection(editor: dict | None) -> tuple[Image.Image, Box]:
    if not editor or editor.get("background") is None:
        raise ValueError("방 사진을 올려 주세요.")
    room = clean_image(editor["background"])
    bounds = brush_mask(editor, room.size).getbbox()
    if bounds is None:
        raise ValueError("가구가 차지할 영역을 방 사진에 칠해 주세요.")
    x0, y0, x1, y1 = bounds
    if x1 - x0 < max(8, room.width * .03) or y1 - y0 < max(8, room.height * .03):
        raise ValueError("배치 영역이 너무 작습니다. 가구 전체가 들어갈 크기로 칠해 주세요.")
    return room, Box(x0 / room.width, y0 / room.height, x1 / room.width, y1 / room.height)


def placement_guide(room: Image.Image, box: Box) -> Image.Image:
    guide = room.copy()
    x0, y0, x1, y1 = box.pixels(room.size)
    ImageDraw.Draw(guide).rectangle((x0, y0, x1 - 1, y1 - 1),
                                    outline="#ef493e", width=max(2, room.width // 180))
    return guide


def generation_size(size: tuple[int, int], longest: int) -> tuple[int, int]:
    if longest not in (512, 768, 1024):
        raise ValueError("생성 해상도는 512, 768, 1024 중에서 선택해 주세요.")
    scale = longest / max(size)
    return tuple(max(64, round(d * scale / 16) * 16) for d in size)


def editable_mask(size: tuple[int, int], box: Box,
                  protection: Image.Image | None = None) -> Image.Image:
    """그림자용 여백을 포함하고, 경계 안쪽에서만 페더링합니다.

    모델은 마스크 인페인팅을 지원하지 않으므로 이 마스크는 결과 복원에만 사용합니다.
    영역 밖 원본 보존은 코드의 성질이며 모델 품질 지표가 아닙니다.
    """
    w, h = size
    x0, y0, x1, y1 = box.pixels(size)
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_y = max(8, round(bw * .20)), max(8, round(bh * .15))
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + max(pad_y, round(bh * .30)))
    # ponytail: 직사각형 여백. 실험에서 그림자가 잘리면 사용자 마스크/세분화로 확장합니다.
    feather = max(2, min(16, round(min(bw, bh) * .08)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.minimum.reduce([xx - x0 + 1, x1 - xx, yy - y0 + 1, y1 - yy])
    values = np.clip(distance / feather, 0, 1)
    values = .5 - .5 * np.cos(np.pi * values)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = np.rint(values * 255).astype(np.uint8)
    if protection is not None:
        if protection.size != size:
            raise ValueError("보호 영역과 방 사진의 크기가 다릅니다.")
        mask[np.asarray(protection.convert("L")) > 0] = 0
    return Image.fromarray(mask)


def preserve_room(room: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    if mask.size != room.size:
        raise ValueError("편집 마스크와 방 사진의 크기가 다릅니다.")
    raw = np.asarray(generated.convert("RGB").resize(room.size, Image.Resampling.LANCZOS))
    original = np.asarray(room.convert("RGB"))
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255
    output = np.rint(original * (1 - alpha) + raw * alpha).clip(0, 255).astype(np.uint8)
    output[np.asarray(mask) == 0] = original[np.asarray(mask) == 0]
    return Image.fromarray(output)


def layout_map(size: tuple[int, int], box: Box) -> Image.Image:
    image = Image.new("RGB", size, "black")
    x0, y0, x1, y1 = box.pixels(size)
    ImageDraw.Draw(image).rectangle((x0, y0, x1 - 1, y1 - 1), fill="white")
    return image


def prompt_for(box: Box, description: str = "", guidance_mode: str = "text") -> str:
    if len(description) > 400:
        raise ValueError("가구 설명은 400자 이하로 입력해 주세요.")
    l, t, r, b = (round(v * 100) for v in box.as_list())
    center_x = (box.left + box.right) / 2
    side = "left side" if center_x < .36 else "right side" if center_x > .64 else "center"
    guide = {
        "text": "",
        "marker": "Image 3 is the same room with a red rectangle specifying the desired furniture bounds. "
                  "The rectangle is ONLY a guide; do not reproduce any red lines. ",
        "map": "Image 3 is a binary spatial layout map, NOT a photograph. Its white region gives "
               "the desired furniture bounds; black means the unchanged room. Never copy the map "
               "colors, blocks or boundaries into the finished photograph. ",
    }
    if guidance_mode not in guide:
        raise ValueError("위치 안내 방식은 text, marker, map 중 하나여야 합니다.")
    return (
        "Create a photorealistic furniture insertion edit. Image 1 is the original room "
        "and the camera view to preserve. Image 2 is a product reference: insert exactly "
        "ONE piece of this furniture, preserving its recognizable design, material, color "
        "and proportions. " + guide[guidance_mode] +
        f"Position the furniture at the {side} of the room. Its complete visible silhouette "
        f"should fit between x={l}% and x={r}%, y={t}% and y={b}%; "
        f"its floor-contact point is near x={(l+r)/2:g}%, y={b}%. "
        "Adapt the furniture perspective to the room camera. Render realistic ambient "
        "lighting, floor contact, contact shadows and plausible occlusion. Keep the room "
        "layout, walls, windows, floor and existing objects as in Image 1. "
        "Do not reproduce the product photo background. Remove all red lines and guides. "
        "Output only a single finished photo from the original room camera."
        + (f"\nMatch these product features in Image 2: {description.strip()}" if description.strip() else "")
    )
