"""실내의 바닥과 기존 가구 구분. SegFormer B0, 비상업 연구·평가용 모델."""
import cv2
import numpy as np
import torch
from pipeline.depth import device

MODEL_ID = 'nvidia/segformer-b0-finetuned-ade-512-512'
REVISION = '489d5cd81a0b59fab9b7ea758d3548ebe99677da'
_MODEL = _PROCESSOR = None


def floor_regions(room):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        _PROCESSOR = SegformerImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        _MODEL = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, revision=REVISION).eval().to(device())
    inputs = _PROCESSOR(images=room.convert('RGB'), return_tensors='pt').to(device())
    with torch.inference_mode():
        logits = _MODEL(**inputs).logits
        labels = torch.nn.functional.interpolate(logits, size=(room.height, room.width),
                                                  mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy()
    floor = np.isin(labels, [3, 28])  # 모델의 0-based floor, rug
    available = floor.copy()
    # 기존 가구 아래 보이는 바닥을 새 가구가 들어갈 빈자리로 오인하지 않게 한다.
    # ponytail: 2D 가구 박스의 하단 45%를 점유 면적으로 근사한다. 실제 3D 부피는 아니다.
    for label in (7, 10, 15, 19, 23, 30, 33, 45, 56, 64, 69, 75, 110):
        _, _, stats, _ = cv2.connectedComponentsWithStats((labels == label).astype(np.uint8), 8)
        for x, y, w, h, area in stats[1:]:
            if area >= .002 * labels.size:
                available[y+int(h*.55):y+h, x:x+w] = False
    return floor, available
