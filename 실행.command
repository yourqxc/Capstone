#!/bin/zsh
set -eu
cd -- "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "먼저 터미널에서 zsh scripts/setup_mac.sh 를 실행해 주세요."
  read -r "reply?엔터를 누르면 닫습니다. "
  exit 1
fi
exec .venv/bin/python app.py
