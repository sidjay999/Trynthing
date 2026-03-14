# 🧣 Saree Dataset Collection Guide

## Folder Structure
```
data/saree_raw/
├── person/           ← Model WEARING the saree
│   ├── saree_001.jpg
│   └── ...
└── garment/          ← Saree PRODUCT image
    ├── saree_001.jpg   (same number = same saree)
    └── ...
```

## Where to Find
- Amazon.in → search "saree women" → save model photo + product photo
- Myntra / Meesho / Ajio → same pattern

## Rules
- ✅ Full body, front-facing, clean background
- ✅ Same saree in both person + garment image (match by number)
- ✅ Min 200 pairs
- ❌ No cropped, blurry, side-angle, or multi-person images

## After Collecting (WSL commands)
```bash
cd /mnt/c/Users/JAY/OneDrive/Desktop/trynthing

# Organize + resize
PYTHONPATH=CatVTON:. ~/miniconda3/bin/python scripts/scrape_saree.py organize \
    --input data/saree_raw/ --output data/saree/

# Generate masks
PYTHONPATH=CatVTON:. ~/miniconda3/bin/python scripts/prepare_dataset.py masks \
    --dataset data/saree/train/ --cloth-type overall

# Validate
PYTHONPATH=CatVTON:. ~/miniconda3/bin/python scripts/prepare_dataset.py validate \
    --dataset data/saree/train/
```
