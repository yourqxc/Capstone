"""Run FLUX.2 Klein on this computer; no hosted inference endpoint is used."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import os
import platform
import re
import subprocess
import sys
import tempfile
from threading import Lock

from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MLX_MODEL_ID = "mlx-community/flux2-klein-4b-4bit"
ROOT = Path(__file__).resolve().parent.parent
_GENERATION_LOCK = Lock()


def resolve_backend(backend: str = "auto") -> str:
    if backend not in {"auto", "mlx", "cuda"}:
        raise ValueError("실행 장치는 auto, mlx, cuda 중 하나를 선택해 주세요.")
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if backend == "auto" and apple_silicon:
        return "mlx"
    if backend == "mlx":
        if not apple_silicon:
            raise RuntimeError("MLX 실행에는 Apple Silicon Mac이 필요합니다.")
        return "mlx"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU 런타임을 설치해 주세요. Mac: requirements-mac.txt, NVIDIA: requirements-cuda.txt") from exc
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("NVIDIA GPU를 찾지 못했습니다. Colab에서 런타임 유형을 GPU로 변경해 주세요.")


def _failure_detail(log: str) -> str:
    """Bound upstream diagnostics and hide common Hugging Face token strings."""
    tail = log[-2000:]
    return re.sub(r"hf_[A-Za-z0-9]+", "[토큰 생략]", tail).strip()


def _generate_mlx(reference_paths: list[Path], prompt: str, width: int, height: int, seed: int) -> Image.Image:
    if find_spec("mflux") is None:
        raise RuntimeError(
            f"현재 앱을 실행한 Python에 Mac 런타임(mflux)이 없습니다.\n실행 Python: {sys.executable}\n"
            "프로젝트 폴더에서 .venv/bin/python app.py로 실행해 주세요.\n"
            "프로젝트 가상환경에도 런타임이 없다면 .venv/bin/python -m pip install -r requirements-mac.txt로 설치해 주세요."
        )
    with tempfile.TemporaryDirectory(prefix="roomfit-mlx-") as directory:
        output = Path(directory) / "result.png"
        # Use this interpreter's module, so a different shell/uv environment cannot be selected accidentally.
        command = [
            sys.executable, "-m", "mflux.models.flux2.cli.flux2_edit_generate",
            "--model", MLX_MODEL_ID, "--base-model", "flux2-klein-4b",
            "--image-paths", *(str(path) for path in reference_paths),
            "--prompt", prompt, "--width", str(width), "--height", str(height),
            "--seed", str(seed), "--steps", "4", "--guidance", "1.0",
            "--quantize", "4", "--low-ram", "--output", str(output),
        ]
        environment = os.environ.copy()
        environment.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
        environment.setdefault("MFLUX_CACHE_DIR", str(ROOT / ".cache" / "mflux"))
        # The child exits after every image and releases all model/unified memory.
        with (Path(directory) / "runtime.log").open("w+b") as log:
            try:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("모델 다운로드 또는 생성이 30분을 초과했습니다. 다운로드 상태와 메모리를 확인해 주세요.") from exc
            finally:
                # Keep upstream timing/memory evidence on success, failure, and timeout without retaining a giant log.
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 33 * 1024))
                tail = log.read().decode("utf-8", errors="replace")
                sanitized = re.sub(r"hf_[A-Za-z0-9]+", "[토큰 생략]", tail)
                sanitized = sanitized.encode("utf-8")[-32 * 1024:].decode("utf-8", errors="ignore")
                (reference_paths[0].parent / "runtime.log").write_text(sanitized, encoding="utf-8")
            detail = _failure_detail(sanitized)
        if result.returncode != 0 or not output.is_file():
            if result.returncode < 0 or "out of memory" in detail.lower():
                raise RuntimeError("Mac 모델 실행이 메모리 부족 또는 프로세스 종료로 실패했습니다. 다른 앱을 닫고 512px로 다시 실행해 주세요.")
            if "Failed to load GPU kernel" in detail or "metal::device" in detail:
                raise RuntimeError("MLX가 Mac GPU에 접근하지 못했습니다. 프로젝트의 가상환경을 활성화한 일반 macOS 터미널에서 실행해 주세요. 제한된 샌드박스 안에서는 GPU 접근이 차단될 수 있습니다.")
            raise RuntimeError(f"MLX 모델 실행에 실패했습니다 (코드 {result.returncode}).\n{detail}")
        with Image.open(output) as image:
            return image.convert("RGB")


@lru_cache(maxsize=1)
def _cuda_pipeline():
    try:
        import torch
        from diffusers import Flux2KleinPipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("CUDA 런타임이 없습니다. pip install -r requirements-cuda.txt를 실행해 주세요.") from exc
    quantization = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
        },
        components_to_quantize=["transformer", "text_encoder"],
    )
    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL_ID, dtype=torch.float16, quantization_config=quantization,
    )
    pipeline.enable_model_cpu_offload(device="cuda")
    pipeline.vae.enable_tiling()
    return pipeline


def _generate_cuda(reference_paths: list[Path], prompt: str, width: int, height: int, seed: int) -> Image.Image:
    import torch

    references = []
    for path in reference_paths:
        with Image.open(path) as image:
            reference = image.convert("RGB")
        # Reference tokens also cost memory. Preserve aspect ratios while bounding each reference.
        longest = max(width, height)
        reference.thumbnail((longest, longest), Image.Resampling.LANCZOS)
        references.append(reference)
    try:
        return _cuda_pipeline()(
            image=references, prompt=prompt, width=width, height=height,
            num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0].convert("RGB")
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU 메모리가 부족합니다. 512px로 낮추거나 Colab 런타임을 재시작해 주세요.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"CUDA 모델 실행에 실패했습니다.\n{_failure_detail(str(exc))}") from exc


def generate(reference_paths: list[Path], prompt: str, width: int, height: int, seed: int, backend: str = "auto") -> Image.Image:
    """References are ordered: clean room, furniture, optional marked placement guide."""
    paths = [Path(path).resolve() for path in reference_paths]
    if not 1 <= len(paths) <= 3 or any(not path.is_file() for path in paths):
        raise ValueError("존재하는 참고 이미지 1~3개가 필요합니다.")
    if not prompt.strip() or len(prompt) > 4000:
        raise ValueError("편집 지시문은 1~4000자로 입력해 주세요.")
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 64 or size > 1024 or size % 16 for size in (width, height)):
        raise ValueError("이미지 크기는 64~1024 사이의 16의 배수여야 합니다.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("시드는 0~4294967295 범위의 정수여야 합니다.")
    selected = resolve_backend(backend)
    # ponytail: one model job at a time; separate workers only when memory permits concurrent generation.
    with _GENERATION_LOCK:
        if selected == "mlx":
            return _generate_mlx(paths, prompt, width, height, seed)
        return _generate_cuda(paths, prompt, width, height, seed)
