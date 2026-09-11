#!/usr/bin/env bash
# Creates a virtual environment (.venv) inside this folder and installs the
# dependencies needed to run the data-collection scripts.
#
# Usage (from anywhere):
#   bash automatic_door/setup_venv.sh        # from the repository root
#
# On the Jetson this reuses the global CUDA torch / OpenCV / NumPy via
# --system-site-packages and only adds ultralytics on top.
set -e

# Always work in the folder this script lives in (automatic_door/).
cd "$(dirname "$0")"

echo "Creating .venv (with --system-site-packages) in: $(pwd)"
python3 -m venv .venv --system-site-packages

echo "Upgrading pip and installing requirements..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "Done. Run the collector with:"
echo "  cd $(pwd)/automatic_labeling"
echo "  ../.venv/bin/python data_collection.py"
