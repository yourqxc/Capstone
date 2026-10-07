"""실행 기록 전체를 품질 평가지와 런타임 표로 내보냅니다. 모델을 호출하지 않습니다."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image


def export(runs: Path, output: Path) -> Path:
    manifests = sorted(runs.glob("*/run.json"))
    if not manifests:
        raise ValueError("run.json이 없습니다. 먼저 실제 합성을 실행해 주세요.")
    folder = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    runtime_rows, quality_rows = [], []
    for manifest in manifests:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        run_folder = manifest.parent
        changed_outside = ""
        if meta["status"] == "succeeded":
            with Image.open(run_folder / "room.png") as image:
                room = np.asarray(image.convert("RGB"))
            with Image.open(run_folder / "result.png") as image:
                result = np.asarray(image.convert("RGB"))
            with Image.open(run_folder / "edit_mask.png") as image:
                locked = np.asarray(image.convert("L")) == 0
            if room.shape != result.shape or locked.shape != room.shape[:2]:
                raise ValueError(f"입력·결과·마스크 크기가 다릅니다: {run_folder}")
            changed_outside = int(np.count_nonzero(np.any(room != result, axis=2) & locked))
        runtime_rows.append({
            "run_id": run_folder.name, "status": meta["status"], "backend": meta["backend"],
            "guide": meta.get("guidance_mode", "legacy-record"), "seed": meta["seed"],
            "resolution": "x".join(map(str, meta["generation_size"])),
            "elapsed_seconds_including_loading": meta.get("elapsed_seconds", ""),
            "changed_pixels_outside_edit_mask": changed_outside,
            "room_sha256": meta["input_sha256"]["room.png"],
            "furniture_sha256": meta["input_sha256"]["furniture.png"],
            "error": meta.get("error", ""),
        })
        quality_rows.append({
            "run_id": run_folder.name, "runtime_status": meta["status"],
            "result_path": str(run_folder / "result.png") if meta["status"] == "succeeded" else "",
            "reviewer": "", "naturalness_1to5": "", "identity_1to5": "", "position_1to5": "",
            "contact_shadow_1to5": "", "occlusion_1to5": "", "artifact_present_0or1": "",
            "usable_0or1": "", "comments": "",
        })
    for filename, rows in (("runtime.csv", runtime_rows), ("quality_review.csv", quality_rows)):
        with (folder / filename).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "total_runs": len(manifests),
        "runtime_succeeded": sum(r["status"] == "succeeded" for r in runtime_rows),
        "runtime_failed": sum(r["status"] == "failed" for r in runtime_rows),
        "quality_scores": "not_assessed; quality_review.csv는 빈 사람 평가 양식입니다.",
        "pixel_preservation": "코드 계약이며 자연스러움/제품 동일성 지표가 아닙니다.",
    }
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실패를 포함한 전체 실행의 평가지 생성")
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    try:
        print(export(args.runs, args.output))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
