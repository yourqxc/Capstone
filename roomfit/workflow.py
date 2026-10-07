"""같은 입력·프롬프트·시드·마스크·원시 결과를 보존하는 실행 흐름."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time
import uuid
import zipfile

import numpy as np
from PIL import Image

from roomfit import engine
from roomfit.placement import (Box, clean_image, editable_mask, generation_size, layout_map,
                              placement_guide, preserve_room, prompt_for)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "outputs" / "runs"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(room: Image.Image, furniture: Image.Image, box: Box, *, seed: int = 42,
        longest: int = 512, description: str = "", backend: str = "auto",
        protection: Image.Image | None = None, output_root: Path = RUNS,
        guidance_mode: str = "text") -> dict:
    room, furniture = clean_image(room), clean_image(furniture)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("시드는 0~4294967295 사이 정수여야 합니다.")
    width, height = generation_size(room.size, longest)
    item = furniture.copy()
    item.thumbnail((longest, longest), Image.Resampling.LANCZOS)
    if min(item.size) < 16:
        raise ValueError("가구 사진이 너무 길쭉합니다. 가구 중심으로 잘라 다시 올려 주세요.")
    prompt = prompt_for(box, description, guidance_mode)
    selected_backend = engine.resolve_backend(backend)
    mask = editable_mask(room.size, box, protection)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    room.save(folder / "room.png")
    furniture.save(folder / "furniture.png")
    placement_guide(room, box).save(folder / "placement.png")
    mask.save(folder / "edit_mask.png")
    references = [folder / "reference_room.png", folder / "reference_furniture.png"]
    room.resize((width, height), Image.Resampling.LANCZOS).save(references[0])
    item.save(references[1])
    if guidance_mode != "text":
        references.append(folder / "reference_placement.png")
        guide = layout_map((width, height), box) if guidance_mode == "map" else placement_guide(
            room.resize((width, height), Image.Resampling.LANCZOS), box)
        guide.save(references[2])
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    packages = {}
    for name in ("Pillow", "numpy", "gradio", "mflux", "mlx", "diffusers", "transformers", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    metadata = {
        "schema_version": 1, "created_utc": stamp, "status": "running",
        "python_executable": sys.executable, "python_version": sys.version.split()[0],
        "model": engine.MODEL_ID, "backend": selected_backend,
        "weights": engine.MLX_MODEL_ID if selected_backend == "mlx" else engine.MODEL_ID,
        "quantization": "mlx-4bit" if selected_backend == "mlx" else "bitsandbytes-nf4",
        "package_versions": packages,
        "seed": seed, "steps": 4, "generation_size": [width, height],
        "original_size": list(room.size), "box_normalized": box.as_list(),
        "description": description, "references": [p.name for p in references],
        "guidance_mode": guidance_mode,
        "input_sha256": {name: _hash(folder / name) for name in ("room.png", "furniture.png")},
        "quality_verified": False,
    }
    manifest = folder / "run.json"
    def record():
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    record()
    started = time.perf_counter()
    try:
        raw = engine.generate(references, prompt, width, height, seed, selected_backend)
        raw.save(folder / "raw.png")
        result = preserve_room(room, raw, mask)
        result.save(folder / "result.png")
        locked = np.asarray(mask) == 0
        unchanged = np.all(np.asarray(result)[locked] == np.asarray(room)[locked])
        metadata.update(status="succeeded", elapsed_seconds=round(time.perf_counter() - started, 3),
                        protected_pixels_unchanged=bool(unchanged),
                        editable_fraction=round(float((~locked).mean()), 6))
    except Exception as exc:
        metadata.update(status="failed", elapsed_seconds=round(time.perf_counter() - started, 3),
                        error=str(exc))
        record()
        raise RuntimeError(f"합성을 완료하지 못했습니다.\n{exc}\n실행 기록: {folder}") from exc
    record()
    with zipfile.ZipFile(folder / "evidence.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.iterdir()):
            if path.suffix in (".png", ".txt", ".json", ".log"):
                bundle.write(path, arcname=path.name)
    return {"image": result, "folder": folder, "metadata": metadata,
            "zip": folder / "evidence.zip", "result": folder / "result.png"}
