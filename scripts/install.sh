#!/bin/bash
set -e

echo "=== NVIDIA AI Gateway Installer ==="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3.8+ is required but not found."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ required. Found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "To run the gateway:"
echo "  source venv/bin/activate"
echo "  python -m gateway --port 8080"
echo ""
echo "Or use the run script: ./run.sh"
