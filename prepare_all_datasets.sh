#!/usr/bin/env bash
set -e
PYTHON="${PYTHON:-python}"
if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

"$PYTHON" src/download_datasets.py --datasets ml-1m ml-20m goodbooks

echo "[prepare_all] done."
