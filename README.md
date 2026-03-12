# 👔 AI Virtual Try-On — SaaS Platform

> AI-powered virtual try-on platform that lets fashion sellers create shareable try-on experiences for their customers. Customers scan a QR code, upload a photo, and see themselves wearing the garment — with photorealistic quality.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

---

## ✨ Features

- **Photorealistic Try-On** — Diffusion-based inpainting with body-aware masking (DensePose + SCHP)
- **Seller Dashboard** — Upload garments, manage catalog, generate QR codes
- **Customer Try-On** — Scan QR → upload photo → see result in seconds
- **Privacy-First** — Customer photos processed in-memory, **never stored**
- **Production Quality** — 768×1024 resolution, 50-step inference, bf16 precision
- **Full Body Masking** — DensePose (body segmentation) + SCHP-LIP + SCHP-ATR (3 models combined)

## 🎯 How It Works

```
Seller uploads garment image
    → Platform generates unique try-on link + QR code
        → Customer scans QR, uploads their photo
            → AI generates photorealistic try-on result
                → Customer sees result (photo deleted from memory)
```

## 🖼️ Try-On Pipeline

| Component | Purpose |
|---|---|
| **DensePose** | Precise body part segmentation (torso, arms, legs) |
| **SCHP-LIP** | Semantic human parsing (20 body part labels) |
| **SCHP-ATR** | Attribute-based parsing (clothing segmentation) |
| **AutoMasker** | Combines all 3 models for cloth-agnostic mask |
| **SD-Inpainting** | Stable Diffusion inpainting backbone |
| **CatVTON Attention** | Concatenation-based attention for garment transfer |

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with 6GB+ VRAM (tested on RTX 4050)
- **Linux / WSL2** (DensePose requires Linux — see WSL setup below)
- **Python 3.9**
- **CUDA 12.1+**

### Installation

```bash
# Clone the repository
git clone https://github.com/sidjay999/Trynthing.git
cd Trynthing

# Create conda environment
conda create -n tryon python=3.9 -y
conda activate tryon

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install detectron2 (required for DensePose — Linux/WSL only)
pip install 'git+https://github.com/facebookresearch/fvcore.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install all dependencies
pip install -r requirements.txt
```

### Launch the Platform

```bash
PYTHONPATH=CatVTON:. python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Windows (WSL2) Setup

If you're on Windows, run the platform inside WSL2 for DensePose support:

```bash
# Enter WSL
wsl -d Ubuntu

# Navigate to project
cd /mnt/c/path/to/ai-virtual-tryon

# Launch server
PYTHONPATH=CatVTON:. python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

Access from Windows browser at `http://localhost:8000`.

## 📁 Project Structure

```
├── CatVTON/                  # Core model (attention, pipeline, masking)
│   ├── model/
│   │   ├── pipeline.py       # CatVTON diffusion pipeline
│   │   ├── cloth_masker.py   # AutoMasker (DensePose + SCHP)
│   │   ├── SCHP/             # Human parsing model
│   │   └── DensePose/        # Body segmentation model
│   ├── densepose/            # DensePose inference code
│   └── utils.py              # Image processing utilities
│
├── app/                      # SaaS platform
│   ├── api/server.py         # FastAPI backend + GPU inference engine
│   ├── templates/            # HTML frontend (seller, customer, landing)
│   └── preprocessing/        # Garment segmentation, pose detection
│
├── scripts/                  # Utilities
│   ├── run_tryon.py          # CLI try-on inference
│   ├── train_lora.py         # LoRA fine-tuning script
│   ├── prepare_dataset.py    # Dataset preparation pipeline
│   └── download_dataset.py   # VITON-HD dataset downloader
│
├── configs/                  # GPU/training configurations
├── requirements.txt          # All dependencies
└── README.md
```

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| Resolution | 768×1024 | Output image size |
| Steps | 50 | Diffusion inference steps |
| Guidance Scale | 2.5 | Classifier-free guidance |
| Precision | bf16 | Mixed precision (bfloat16) |
| Seed | 42 | Reproducibility seed |

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/seller/upload` | Upload garment image |
| `GET` | `/api/seller/garments` | List all garments |
| `GET` | `/api/seller/qr/{id}` | Get QR code for garment |
| `POST` | `/api/tryon/{id}` | Run try-on (customer) |
| `DELETE` | `/api/seller/garments/{id}` | Delete garment |

## 🧠 LoRA Fine-Tuning

Train on custom garment pairs for better results:

```bash
# Download VITON-HD dataset
python scripts/download_dataset.py

# Fine-tune with LoRA
PYTHONPATH=CatVTON:. python scripts/train_lora.py \
    --data_dir data/train \
    --output_dir models/lora \
    --num_steps 2000 \
    --learning_rate 1e-4
```

## 🌐 Deployment (ngrok — Self-Hosted)

Run the platform on your own GPU and expose it publicly with ngrok:

### 1. Install ngrok

```bash
# Linux/WSL
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.tgz | tar xz
sudo mv ngrok /usr/local/bin/

# Sign up at https://ngrok.com (free) and add your auth token:
ngrok config add-authtoken YOUR_TOKEN
```

### 2. Start the server

```bash
# Terminal 1 — Start the try-on server
cd /mnt/c/Users/JAY/OneDrive/Desktop/trynthing
PYTHONPATH=CatVTON:. ~/miniconda3/bin/python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

### 3. Expose publicly

```bash
# Terminal 2 — Get a public URL
ngrok http 8000
```

ngrok will show a public URL like `https://abc123.ngrok-free.app` — share this with anyone to try on garments.

## 📊 Performance

| Metric | Value |
|---|---|
| Inference Time | ~60s (RTX 4050 6GB) |
| VRAM Usage | ~4-5 GB |
| Resolution | 768×1024 |
| Model Size | ~2.5 GB (auto-downloaded) |

## 🛣️ Roadmap

- [x] Core try-on pipeline with CatVTON
- [x] Full AutoMasker (DensePose + SCHP-LIP + SCHP-ATR)
- [x] SaaS platform with seller/customer flow
- [x] QR code generation & privacy-first design
- [ ] Saree & full-body garment support
- [ ] LoRA fine-tuning on custom datasets
- [ ] Production cloud deployment
- [ ] Multi-garment try-on (accessories, shoes)

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

This project builds upon:
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) (CreativeML Open RAIL-M License)
- [DensePose](https://github.com/facebookresearch/detectron2) (Apache 2.0)
- [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) (MIT License)

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- New garment type support (saree, lehenga, etc.)
- Performance optimizations
- Cloud deployment templates
