#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-core}"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .

if [[ "$MODE" == "gpu" ]]; then
  pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
  pip install 'fire>=0.5'
  pip install simple-lama-inpainting==0.1.2 --no-deps
elif [[ "$MODE" == "cpu-ai" ]]; then
  pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
  pip install 'fire>=0.5'
  pip install simple-lama-inpainting==0.1.2 --no-deps
fi

echo "Installed. Activate with: source .venv/bin/activate"
echo "Or run directly: ./wmclean.sh clean ./pages -o ./cleaned --backend auto"
