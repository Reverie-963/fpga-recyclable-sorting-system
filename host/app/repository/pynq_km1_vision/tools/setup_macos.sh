#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-yolo
.venv-yolo/bin/python -m pip install --upgrade pip
.venv-yolo/bin/python -m pip install torch torchvision
.venv-yolo/bin/python -m pip install -r requirements_yolo.txt
.venv-yolo/bin/python -m pip check
.venv-yolo/bin/python -m src.doctor
