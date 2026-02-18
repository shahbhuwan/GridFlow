#!/bin/bash
set -e

# ======================================================
#  GridFlow CLI Environment Setup (macOS/Linux)
# ======================================================

echo "======================================================"
echo " GridFlow CLI Environment Setup"
echo "======================================================"
echo ""

# --- Detect a compatible Python interpreter ---
# Tier 1: Try "python3" on PATH
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "[CRITICAL ERROR] Python not found."
    echo "GridFlow CLI requires Python 3.11 or newer."
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# Check version
$PYTHON_CMD -c "import sys; exit(0 if (3, 11) <= sys.version_info < (3, 15) else 1)" || {
    echo ""
    echo "[CRITICAL ERROR] Compatible Python not found."
    echo "GridFlow CLI requires Python 3.11, 3.12, 3.13, or 3.14."
    echo "(Python 3.15+ is not yet supported)."
    echo "Detected: $($PYTHON_CMD --version)"
    exit 1
}

echo "Detected Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# --- Create Virtual Environment ---
VENV_NAME="gridflow_env"

# Allow user override
read -p "Install Path (Leave empty for default './$VENV_NAME'): " USER_VENV_PATH
if [ -n "$USER_VENV_PATH" ]; then
    VENV_NAME="$USER_VENV_PATH"
fi

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists."
    echo "Attempting to update/use existing environment..."
else
    echo "Creating virtual environment at '$VENV_NAME'..."
    $PYTHON_CMD -m venv "$VENV_NAME" || {
        echo "ERROR: Failed to create virtual environment."
        exit 1
    }
fi

echo ""
echo "--- Activating virtual environment ---"
source "$VENV_NAME/bin/activate"

echo ""
echo "--- Installing Dependencies ---"
pip install --upgrade pip
pip install -r requirements_cli.txt
# Optional: pip install -r requirements_gui.txt

echo ""
echo "--- Installing GridFlow ---"
pip install .

echo ""
echo "======================================================"
echo " Setup Complete!"
echo "======================================================"
echo ""
echo "To use the GridFlow CLI, you must activate this environment."
echo "Run the following command:"
echo ""
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Once activated, you can use the 'gridflow' command."
echo ""
