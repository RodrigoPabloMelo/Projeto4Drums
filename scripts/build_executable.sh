#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="${APP_NAME:-Drumble}"
PY_BIN="${PY_BIN:-.venv/bin/python}"
BUILD_MODE="${BUILD_MODE:-onedir}" # onedir | onefile
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Python do venv nao encontrado em: $PY_BIN"
  echo "Crie o ambiente primeiro: uv sync"
  exit 1
fi

if ! "$PY_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller nao encontrado na venv. Instalando..."
  UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --python "$PY_BIN" pyinstaller
fi

echo "Limpando build anterior..."
rm -rf build dist

PYI_ARGS=(
  --noconfirm
  --clean
  --name "$APP_NAME"
  --windowed
  --add-data "assets:assets"
  --add-data "sounds:sounds"
  --collect-all mediapipe
  --collect-all pygame
  --collect-all cv2
  main.py
)

if [[ "$BUILD_MODE" == "onefile" ]]; then
  PYI_ARGS=(--onefile "${PYI_ARGS[@]}")
else
  PYI_ARGS=(--onedir "${PYI_ARGS[@]}")
fi

echo "Gerando executavel (${BUILD_MODE})..."
"$PY_BIN" -m PyInstaller "${PYI_ARGS[@]}"

if [[ "$BUILD_MODE" == "onefile" ]]; then
  echo "Build concluido: $ROOT_DIR/dist/$APP_NAME"
else
  echo "Build concluido: $ROOT_DIR/dist/$APP_NAME/$APP_NAME"
fi
