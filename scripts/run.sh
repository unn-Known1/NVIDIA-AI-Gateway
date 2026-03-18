#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running installation..."
    ./scripts/install.sh
    source venv/bin/activate
fi

# Activate venv
source venv/bin/activate

# Set default port if not provided
PORT=${1:-8080}

echo "Starting NVIDIA AI Gateway on port $PORT..."
python -m gateway --port "$PORT"
