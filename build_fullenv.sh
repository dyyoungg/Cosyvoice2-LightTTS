#!/usr/bin/env bash
set -euo pipefail

# 构建全环境自包含可执行程序
# 使用本机 Python/虚拟环境进行构建，并将依赖收集到 dist/cosyvoice_fullenv

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "[1/3] 清理旧产物"
rm -rf build/ dist/

echo "[2/3] 使用 fullenv.spec 构建 (ONEDIR)"
pyinstaller fullenv.spec --clean --noconfirm --log-level=INFO

echo "[3/3] 可执行文件位置: $PROJECT_ROOT/dist/cosyvoice_fullenv/cosyvoice_fullenv"
echo "运行示例: ./dist/cosyvoice_fullenv/cosyvoice_fullenv --model_dir /path/to/model_dir"


