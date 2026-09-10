"""홀드아웃 검증 — 바닥 추정 IoU가 일반화되는가.

## 왜 필요한가

`samples/rooms/`의 10장은 **tol을 고르는 데 쓴 바로 그 방들**이다. 같은 집합으로
튜닝하고 같은 집합으로 평가했으므로 IoU 0.864는 일반화 성능이 아니다.
심사에서 "0.864는 몇 장에 대한 값이고 파라미터는 어떻게 정했나"는 질문 하나로 무너진다.

그래서 **튜닝에 한 번도 쓰지 않은 방 10장**을 같은 선정 조건으로 뽑아 다시 잰다.
tol은 0.022로 고정하고 이 방들로는 아무것도 조정하지 않는다.

선정 조건(튜닝셋과 동일): ADE20K validation, 씬 {bedroom, living_room, home_office,
dorm_room}, 가로 사진, 정답 마스크 기준 바닥 >= 10% / 벽 >= 12%, 가로 600px 이상.
기존 10장을 제외하면 55장이 통과하고 그중 상위 10장을 썼다.

    python eval/holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.depth import estimate_depth  # noqa: E402
from pipeline.geometry import floor_plane, iou  # noqa: E402

TUNING = {"평균": 0.864, "중앙값": 0.877, "최저": 0.679}


def main():
    rooms = sorted((ROOT / "samples" / "holdout").glob("*.png"))
    if not rooms:
        sys.exit("samples/holdout 이 비어 있습니다.")

    print(f"홀드아웃 {len(rooms)}장 — tol 기본값 고정, 이 방들로는 아무 조정도 하지 않았다\n")
    print(f"{'파일':<14}{'IoU':>7}{'추정 바닥':>10}{'정답 바닥':>10}{'지평선':>8}")
    vals = []
    for p in rooms:
        room = Image.open(p).convert("RGB")
        gt = np.array(Image.open(ROOT / "samples" / "holdout_gt" / f"{p.stem}_gt.png")) == 255
        pl = floor_plane(room, estimate_depth(room))
        v = iou(pl["mask"], gt)
        vals.append(v)
        hy = pl["horizon_y"]
        print(f"{p.stem:<14}{v:7.3f}{pl['mask'].mean() * 100:9.1f}%{gt.mean() * 100:9.1f}%"
              + (f"{hy:8.0f}" if hy is not None else f"{'-':>8}"))

    m, med, lo = float(np.mean(vals)), float(np.median(vals)), float(min(vals))
    print(f"\n{'홀드아웃':<12}평균 {m:.3f}  중앙값 {med:.3f}  최저 {lo:.3f}  최고 {max(vals):.3f}")
    print(f"{'튜닝셋':<12}평균 {TUNING['평균']:.3f}  중앙값 {TUNING['중앙값']:.3f}  "
          f"최저 {TUNING['최저']:.3f}")
    print(f"{'차이':<12}평균 {m - TUNING['평균']:+.3f}  중앙값 {med - TUNING['중앙값']:+.3f}")
    print(f"\n0.78 이상: {sum(v >= 0.78 for v in vals)}/{len(vals)}")

    worst = int(np.argmin(vals))
    rest = [v for i, v in enumerate(vals) if i != worst]
    print(f"최저 사례({rooms[worst].stem}) 제외 시 평균 {np.mean(rest):.3f} "
          f"(튜닝셋 대비 {np.mean(rest) - TUNING['평균']:+.3f})")


if __name__ == "__main__":
    main()
