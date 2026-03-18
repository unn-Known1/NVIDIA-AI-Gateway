#!/bin/bash
set -e

echo "=== Building NVIDIA AI Gateway ==="
echo ""

pip install --upgrade pip > /dev/null 2>&1
pip install pyinstaller > /dev/null 2>&1

echo "Building executable..."
pyinstaller --onefile \
    --name "nvidia-ai-gateway" \
    src/gateway/__main__.py --Distpath ./dist 2>/dev/null

echo ""
echo "Build complete! Executable: dist/nvidia-ai-gateway"
