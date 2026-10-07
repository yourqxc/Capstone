#!/bin/zsh
set -eu
cd -- "$(dirname "$0")/.."
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "이 설치 스크립트는 Apple Silicon Mac용입니다. NVIDIA GPU는 requirements-cuda.txt를 사용해 주세요."
  exit 1
fi
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12가 필요합니다. 설치한 뒤 다시 실행해 주세요."
  exit 1
fi
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac.txt
echo "설치 완료. .venv/bin/python app.py 로 실행해 주세요."
