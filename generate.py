"""UI 없이 동일한 실험을 재현합니다. python generate.py --help"""
import argparse
from pathlib import Path

from PIL import Image

from roomfit.placement import Box
from roomfit.workflow import run


def main():
    parser = argparse.ArgumentParser(description="RoomFit 로컬 가구 합성 및 실행 증거 저장")
    parser.add_argument("--room", type=Path, required=True)
    parser.add_argument("--furniture", type=Path, required=True)
    parser.add_argument("--box", type=float, nargs=4, required=True,
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), help="0~1 정규화 코너 좌표")
    parser.add_argument("--resolution", type=int, choices=(512, 768, 1024), default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--description", default="")
    parser.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    parser.add_argument("--guide", choices=("text", "map", "marker"), default="text",
                        help="실험용 위치 안내 방식. marker는 안내선이 결과에 남을 수 있습니다.")
    args = parser.parse_args()
    try:
        with Image.open(args.room) as image:
            room = image.copy()
        with Image.open(args.furniture) as image:
            furniture = image.copy()
        result = run(room, furniture, Box(*args.box), longest=args.resolution,
                     seed=args.seed, description=args.description, backend=args.backend,
                     guidance_mode=args.guide)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"결과: {result['result']}\n실행 기록: {result['folder']}\n증거 ZIP: {result['zip']}")


if __name__ == "__main__":
    main()
