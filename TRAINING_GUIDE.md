# 🧠 Training & Fine-Tuning Guide — Remote GPU Desktop

> This file contains **every command** you need to run on the remote GPU desktop.
> Copy this file to the remote machine and follow step by step.

---

## Prerequisites on Remote Machine
- **GPU**: A100 / V100 / A40 (16GB+ VRAM recommended)
- **OS**: Ubuntu / Linux
- **CUDA**: 11.8+ installed
- **Storage**: ~20GB free (models + datasets)

---

## Step 0 — Clone & Setup Environment

```bash
# Clone your repo
git clone https://github.com/sidjay999/Trynthing.git
cd Trynthing

# Install Miniconda (if not present)
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
export PATH=~/miniconda3/bin:$PATH

# Create environment
conda create -n tryon python=3.9 -y
conda activate tryon

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install detectron2 + fvcore (for DensePose)
pip install 'git+https://github.com/facebookresearch/fvcore.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install all project dependencies
pip install -r requirements.txt

# Pin critical version
pip install huggingface_hub==0.25.2

# Extra training deps
pip install peft datasets tqdm
```

---

## Step 1 — Download Datasets

### 1a. Download DressCode (full-body baseline, ~5GB)
```bash
cd Trynthing
PYTHONPATH=CatVTON:. python scripts/scrape_saree.py hf --output data/dresscode_raw/ --limit 2000
```

Or manually via HuggingFace:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mattmdjaga/dresscode_1024', local_dir='data/dresscode_raw', repo_type='dataset')
"
```

### 1b. Organize into train/val
```bash
PYTHONPATH=CatVTON:. python scripts/scrape_saree.py organize \
    --input data/dresscode_raw/ --output data/dresscode/
```

### 1c. Download/Copy Saree Dataset
If you scraped sarees on your local machine, copy the `data/saree_raw/` folder to the remote machine, then:
```bash
PYTHONPATH=CatVTON:. python scripts/scrape_saree.py organize \
    --input data/saree_raw/ --output data/saree/
```

Or scrape directly on the remote machine:
```bash
pip install requests beautifulsoup4
PYTHONPATH=CatVTON:. python scripts/scrape_saree.py amazon --pages 15 --output data/saree_raw/
PYTHONPATH=CatVTON:. python scripts/scrape_saree.py organize \
    --input data/saree_raw/ --output data/saree/
```

---

## Step 2 — Generate Masks (GPU needed)

```bash
# Generate cloth-agnostic masks for DressCode
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py masks \
    --dataset data/dresscode/train/ --cloth-type overall

# Generate masks for saree data
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py masks \
    --dataset data/saree/train/ --cloth-type overall

# Validate both datasets
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py validate --dataset data/dresscode/train/
PYTHONPATH=CatVTON:. python scripts/prepare_dataset.py validate --dataset data/saree/train/
```

---

## Step 3 — LoRA Fine-Tuning

### 3a. Train on DressCode first (full-body baseline)
```bash
PYTHONPATH=CatVTON:. python scripts/train_lora.py \
    --dataset data/dresscode/ \
    --cloth-type overall \
    --epochs 5 \
    --batch-size 2 \
    --lr 1e-4 \
    --lora-rank 8 \
    --lora-alpha 8 \
    --grad-accum 4 \
    --height 1024 \
    --width 768 \
    --save-every 2 \
    --checkpoint-dir checkpoints/dresscode/
```

**Expected**: ~3-4 hours on A100, loss should drop below 0.05

### 3b. Continue training on Saree data
```bash
PYTHONPATH=CatVTON:. python scripts/train_lora.py \
    --dataset data/saree/ \
    --cloth-type overall \
    --epochs 20 \
    --batch-size 2 \
    --lr 5e-5 \
    --lora-rank 8 \
    --lora-alpha 8 \
    --grad-accum 4 \
    --height 1024 \
    --width 768 \
    --save-every 5 \
    --resume checkpoints/dresscode/lora_epoch5/ \
    --checkpoint-dir checkpoints/saree/
```

**Expected**: ~1-2 hours on A100 for 500 pairs

---

## Step 4 — Test Results (Quick Verify)

```bash
# Run a test try-on with the fine-tuned LoRA
PYTHONPATH=CatVTON:. python scripts/run_tryon.py \
    --person data/saree/val/person/saree_0000_person.jpg \
    --garment data/saree/val/garment/saree_0000_garment.jpg \
    --cloth-type overall \
    --lora-path checkpoints/saree/lora_epoch20/ \
    --output data/output/saree_test.jpg
```

---

## Step 5 — Download Checkpoints to Local Machine

After training, download the LoRA checkpoints to your local machine:

```bash
# On remote machine — zip checkpoints
cd Trynthing
tar -czf checkpoints_saree.tar.gz checkpoints/saree/

# On your local machine — download via scp
scp remote_user@remote_ip:~/Trynthing/checkpoints_saree.tar.gz .
tar -xzf checkpoints_saree.tar.gz -C c:/Users/JAY/OneDrive/Desktop/trynthing/
```

---

## Step 6 — Integrate LoRA into Server (Back on Local Machine)

Once you have the checkpoints on your local machine:

```bash
# In WSL
cd /mnt/c/Users/JAY/OneDrive/Desktop/trynthing

# Start server with LoRA weights
PYTHONPATH=CatVTON:. HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    ~/miniconda3/bin/python -m uvicorn app.api.server:app \
    --host 0.0.0.0 --port 8000
```

> The server will need code changes to load LoRA weights — those changes will be made before training starts.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `cached_download` import error | `pip install huggingface_hub==0.25.2` |
| `EncoderDecoderCache` missing | `pip install transformers==4.44.2` |
| `LocalEntryNotFoundError` missing | `pip install huggingface_hub==0.25.2` |
| `clear_device_cache` missing | `pip install accelerate>=1.0.0` |
| CUDA out of memory | Reduce `--batch-size` to 1, increase `--grad-accum` |
| torchvision C++ ops error | `pip install torchvision --force-reinstall` |

## Compatible Package Versions

```
huggingface_hub==0.25.2
diffusers==0.27.2
transformers==4.44.2
accelerate==1.10.1
torch (latest with CUDA)
torchvision (matching torch version)
peft>=0.7.0
datasets>=2.14.0
```
