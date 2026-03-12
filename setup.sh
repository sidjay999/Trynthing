#!/bin/bash
# One-command setup for AI Virtual Try-On platform
# Run this on Linux/WSL2 with NVIDIA GPU
set -e

echo "=== AI Virtual Try-On — Setup ==="

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ~/miniconda3
    export PATH=~/miniconda3/bin:$PATH
fi

PYTHON=~/miniconda3/bin/python
PIP=~/miniconda3/bin/pip

echo "=== Installing PyTorch + CUDA ==="
$PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing DensePose dependencies ==="
$PIP install 'git+https://github.com/facebookresearch/fvcore.git'
$PIP install 'git+https://github.com/facebookresearch/detectron2.git'

echo "=== Installing project dependencies ==="
$PIP install -r requirements.txt

echo ""
echo "=== Setup Complete! ==="
echo "Launch with:"
echo "  cd $(pwd)"
echo "  PYTHONPATH=CatVTON:. $PYTHON -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then open http://localhost:8000"
