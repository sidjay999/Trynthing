# 🧠 Training & Fine-Tuning Guide — Remote GPU Desktop

> Complete standalone guide. Copy this file + the repo to your remote GPU machine.
> Follow step by step.

---

## What We're Training
- **Base model**: CatVTON (already handles upper-body try-on well)
- **Goal**: LoRA fine-tune for **saree / full-body** try-on
- **Dataset**: 238 saree pairs (person wearing saree + saree product image)
- **Cloth type**: `overall` (full-body)

---

## Prerequisites on Remote Machine
- **GPU**: A100 / V100 / A40 (16GB+ VRAM)
- **OS**: Ubuntu / Linux
- **CUDA**: 11.8+
- **Storage**: ~20GB free

---

## Step 0 — Clone & Setup

```bash
# Clone repo (includes dataset)
git clone https://github.com/sidjay999/Trynthing.git
cd Trynthing

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
export PATH=~/miniconda3/bin:$PATH

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install detectron2 + fvcore
pip install 'git+https://github.com/facebookresearch/fvcore.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install project deps
pip install -r requirements.txt

# Pin critical versions
pip install huggingface_hub==0.25.2 transformers==4.44.2 accelerate==1.10.1

# Training extras
pip install peft datasets tqdm
```

---

## Step 1 — Prepare Dataset

The saree images are already in the repo at `data/saree/`.
If masks weren't generated yet, run:

```bash
# Generate full-body masks (uses GPU, ~10 min)
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py masks \
    --dataset data/saree/train/ --cloth-type overall

# Generate pairs
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py pairs \
    --dataset data/saree/train/ --strategy matched

# Validate
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py validate \
    --dataset data/saree/train/
```

---

## Step 2 — LoRA Fine-Tuning

```bash
PYTHONPATH=CatVTON:. python scripts/train_lora.py \
    --dataset data/saree/ \
    --cloth-type overall \
    --epochs 20 \
    --batch-size 2 \
    --lr 1e-4 \
    --lora-rank 8 \
    --lora-alpha 8 \
    --grad-accum 4 \
    --height 1024 \
    --width 768 \
    --save-every 5 \
    --checkpoint-dir checkpoints/saree/
```

**Expected**: ~1-2 hours on A100 for 238 pairs × 20 epochs.
Loss should drop below 0.05.

---

## Step 3 — Test (Quick Verify)

```bash
PYTHONPATH=CatVTON:. python scripts/run_tryon.py \
    --person data/saree/val/person/saree_0001.jpg \
    --garment data/saree/val/garment/saree_0001.jpg \
    --cloth_type overall \
    --lora-path checkpoints/saree/lora_epoch20/ \
    --output data/output/saree_test.jpg
```

---

## Step 4 — Download Checkpoints

```bash
# On remote machine
cd Trynthing
tar -czf checkpoints_saree.tar.gz checkpoints/saree/

# On local machine (PowerShell) — download via scp
scp remote_user@remote_ip:~/Trynthing/checkpoints_saree.tar.gz .
tar -xzf checkpoints_saree.tar.gz -C c:\Users\JAY\OneDrive\Desktop\trynthing\
```

---

## Step 5 — Run Server with LoRA (Local Machine, WSL)

```bash
cd /mnt/c/Users/JAY/OneDrive/Desktop/trynthing
PYTHONPATH=CatVTON:. HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    ~/miniconda3/bin/python -m uvicorn app.api.server:app \
    --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `cached_download` error | `pip install huggingface_hub==0.25.2` |
| `EncoderDecoderCache` missing | `pip install transformers==4.44.2` |
| `LocalEntryNotFoundError` missing | `pip install huggingface_hub==0.25.2` |
| torchvision C++ ops error | `pip install torchvision --force-reinstall` |
| CUDA OOM | Reduce `--batch-size 1`, increase `--grad-accum 8` |

## Compatible Versions
```
huggingface_hub==0.25.2
diffusers==0.27.2
transformers==4.44.2
accelerate==1.10.1
peft>=0.7.0
```
