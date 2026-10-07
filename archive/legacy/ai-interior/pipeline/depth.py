"""방 사진의 상대 깊이맵을 추정한다 (Depth Anything V2).

기획서 7쪽 "Depth Map 추출", 8쪽 "Depth Estimation" 대응.

출력은 상대 깊이다. 절대 거리(미터)가 아니라 역깊이(disparity)에 비례하는 값이며,
값이 클수록 카메라에 가깝다. geometry.py가 이 성질을 이용해 바닥 평면을 찾는다.

단독 실행:
    python pipeline/depth.py                      # samples/rooms 전체
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_PIPE = None


def device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load():
    """모델은 처음 쓸 때 한 번만 올린다 (로드 약 7초, 이후 추론 1.3초)."""
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline

        _PIPE = pipeline("depth-estimation", model=MODEL_ID, device=device())
    return _PIPE


def estimate_depth(room: Image.Image) -> np.ndarray:
    """상대 깊이맵 (H, W) float32, 0~1 정규화. 값이 클수록 가깝다.

    **`predicted_depth`(float32 원본)를 쓴다. `depth` 키가 아니다.**
    파이프라인이 돌려주는 `depth`는 min-max 정규화 후 uint8로 양자화한 **시각화용 이미지**라
    고유값이 231~256개뿐이다. 원본은 30만 개가 넘는다.
    바닥 평면 RANSAC의 허용오차가 0.02 수준이라 1/255=0.0039 계단 위에서 맞추면
    임계값 4~9단계 폭 안에서 튜닝하는 셈이 된다.
    """
    out = _load()(room.convert("RGB"))
    raw = out["predicted_depth"]
    d = np.squeeze(raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw))
    d = d.astype(np.float32)
    if d.shape != (room.size[1], room.size[0]):     # 모델 해상도로 나오면 되돌린다
        import cv2

        d = cv2.resize(d, room.size, interpolation=cv2.INTER_LINEAR)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return (d - lo) / (hi - lo)


def colorize(depth: np.ndarray) -> Image.Image:
    """깊이맵을 눈으로 보기 위한 컬러맵 (디버깅·보고서용)."""
    import cv2

    u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return Image.fromarray(bgr[:, :, ::-1])


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or sorted(Path("samples/rooms").glob("*.png"))
    out_dir = Path("samples/rooms_depth")
    out_dir.mkdir(exist_ok=True)
    print(f"device={device()}  대상 {len(targets)}장")
    for p in targets:
        d = estimate_depth(Image.open(p))
        colorize(d).save(out_dir / f"{p.stem}_depth.png")
        np.save(out_dir / f"{p.stem}_depth.npy", d.astype(np.float32))
        print(f"  {p.name:<14} {d.shape}  하단평균 {d[-d.shape[0]//5:].mean():.2f}  상단평균 {d[:d.shape[0]//5].mean():.2f}")
    print(f"저장 위치: {out_dir}")
