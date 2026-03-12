#!/bin/bash
# All-in-one WSL setup for CatVTON with full DensePose support
set -e

export PATH=~/miniconda3/bin:$PATH

echo "=== Step 1: Create conda env ==="
conda create -n catvton python=3.9 -y || echo "Env may already exist"
PYBIN=~/miniconda3/envs/catvton/bin/python
PIPBIN=~/miniconda3/envs/catvton/bin/pip

echo "=== Step 2: Verify Python ==="
$PYBIN --version

echo "=== Step 3: Install PyTorch + CUDA ==="
$PIPBIN install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 4: Install fvcore ==="
$PIPBIN install 'git+https://github.com/facebookresearch/fvcore.git'

echo "=== Step 5: Install detectron2 ==="
$PIPBIN install 'git+https://github.com/facebookresearch/detectron2.git'

echo "=== Step 6: Install diffusers + transformers ==="
$PIPBIN install diffusers==0.27.2 transformers accelerate==0.31.0

echo "=== Step 7: Install numpy + vision libs ==="
$PIPBIN install numpy==1.26.4 opencv-python-headless mediapipe Pillow

echo "=== Step 8: Install ML extras ==="
$PIPBIN install peft datasets xformers

echo "=== Step 9: Install web/API libs ==="
$PIPBIN install fastapi uvicorn python-multipart "qrcode[pil]" aiofiles jinja2

echo "=== Step 10: Install huggingface_hub ==="
$PIPBIN install huggingface_hub

echo "=== Step 11: Verify torch+CUDA ==="
$PYBIN -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== Step 12: Test DensePose import ==="
cd /mnt/c/Users/JAY/OneDrive/Desktop/trynthing
PYTHONPATH=CatVTON:. $PYBIN -c "from model.DensePose import DensePose; print('DensePose: OK!')" || echo "DensePose import failed (will debug)"

echo "=== SETUP COMPLETE ==="
