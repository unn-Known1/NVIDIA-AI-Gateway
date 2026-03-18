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

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."

    # Try creating venv with --without-pip first to avoid ensurepip issues
    if python3 -m venv --without-pip venv 2>/dev/null; then
        echo "✓ Virtual environment created (without pip)"
    else
        echo "Warning: venv creation with --without-pip failed, trying standard method..."
        python3 -m venv venv || {
            echo "ERROR: Failed to create virtual environment."
            echo "Attempting fallback with virtualenv package..."

            # Install virtualenv globally and use it
            python3 -m pip install --user virtualenv || true
            python3 -m virtualenv venv || {
                echo "ERROR: All venv creation methods failed."
                echo "Please install virtualenv manually: pip install virtualenv"
                exit 1
            }
        }
    fi
fi

# Activate venv
source venv/bin/activate

# Ensure pip is available
if ! command -v pip &> /dev/null; then
    echo "Installing pip in virtual environment..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python - || {
        echo "ERROR: Failed to install pip."
        exit 1
    }
fi

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "To run the gateway:"
echo "  source venv/bin/activate"
echo "  python -m gateway --port 8080"
echo ""
echo "Or use the run script: ./run.sh"
