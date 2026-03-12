"""
Dataset Download & Conversion Script for CatVTON LoRA Fine-Tuning

Downloads VITON-HD from HuggingFace and converts to CatVTON format.
Also generates SCHP masks for training.

VITON-HD format (CatVTON native):
    vitonhd/
    ├── test/
    │   ├── image/             # Person images
    │   ├── cloth/             # Garment images  
    │   ├── agnostic-mask/     # Cloth-agnostic masks
    │   └── ...
    └── test_pairs_unpaired.txt

Usage:
    # Download VITON-HD dataset (~207MB)
    python scripts/download_dataset.py vitonhd

    # Download and convert for LoRA training (subset)
    python scripts/download_dataset.py vitonhd --split train --max-pairs 500

    # Create mock saree dataset from augmentation
    python scripts/download_dataset.py saree-mock --num 200

    # Verify downloaded dataset
    python scripts/download_dataset.py verify --path data/vitonhd
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
sys.path.insert(0, str(CATVTON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def download_vitonhd(output_dir: str, split: str = "train", max_pairs: int = None):
    """
    Download VITON-HD dataset from HuggingFace.
    
    The 'Heatmob-Research/VITON-HD' repo has ~4064 rows, ~207MB.
    We also check 'zhengchong/VITON-HD' (CatVTON author's version).
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print(f"  Downloading VITON-HD ({split})")
    print("=" * 50)
    
    # Try the smaller public version first
    print("\n[1/3] Downloading from HuggingFace...")
    
    try:
        # Download using datasets library for structured access
        from datasets import load_dataset
        
        ds = load_dataset("Heatmob-Research/VITON-HD", split=split)
        total = len(ds)
        if max_pairs:
            total = min(total, max_pairs)
        print(f"  Dataset has {len(ds)} samples, using {total}")
        
        # Create directory structure
        img_dir = out / split / "image"
        cloth_dir = out / split / "cloth"
        mask_dir = out / split / "agnostic-mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        cloth_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        pairs = []
        for i in range(total):
            sample = ds[i]
            person_name = f"{i:06d}.jpg"
            cloth_name = f"{i:06d}.jpg"
            
            # Save person image
            if 'image' in sample and sample['image'] is not None:
                sample['image'].save(str(img_dir / person_name))
            
            # Save cloth image
            if 'cloth' in sample and sample['cloth'] is not None:
                sample['cloth'].save(str(cloth_dir / cloth_name))
            
            # Save mask if available
            if 'agnostic_mask' in sample and sample['agnostic_mask'] is not None:
                sample['agnostic_mask'].save(str(mask_dir / person_name.replace('.jpg', '_mask.png')))
            elif 'mask' in sample and sample['mask'] is not None:
                sample['mask'].save(str(mask_dir / person_name.replace('.jpg', '_mask.png')))
            
            pairs.append(f"{person_name} {cloth_name}")
            
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{total}] saved")
        
        # Save pairs file
        pairs_file = out / f"{split}_pairs_unpaired.txt"
        pairs_file.write_text("\n".join(pairs))
        print(f"\n[2/3] Pairs saved: {pairs_file} ({len(pairs)} pairs)")
        
    except Exception as e:
        print(f"  [WARN] HuggingFace datasets failed: {e}")
        print("  Falling back to manual download...")
        _download_vitonhd_manual(out, split, max_pairs)
        return
    
    # Generate masks if missing
    print("\n[3/3] Checking masks...")
    mask_count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
    if mask_count < total:
        print(f"  {mask_count}/{total} masks found, generating remaining...")
        _generate_missing_masks(out / split, total)
    else:
        print(f"  All {mask_count} masks present")
    
    print(f"\n  Dataset ready at: {out}")
    print(f"  Total pairs: {total}")


def _download_vitonhd_manual(output_dir: Path, split: str, max_pairs: int):
    """Fallback: download VITON-HD via snapshot_download."""
    from huggingface_hub import snapshot_download
    
    try:
        path = snapshot_download(
            repo_id="zhengchong/VITON-HD",
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"  [ERROR] Could not download VITON-HD: {e}")
        print("  Please download manually from: https://huggingface.co/datasets/Heatmob-Research/VITON-HD")
        print(f"  And place it in: {output_dir}")


def _generate_missing_masks(split_dir: Path, total: int):
    """Generate SCHP masks for images that don't have them."""
    import torch
    from PIL import Image
    
    img_dir = split_dir / "image"
    mask_dir = split_dir / "agnostic-mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images without masks
    images = sorted(img_dir.glob("*.jpg"))
    missing = []
    for img_path in images:
        mask_path = mask_dir / img_path.name.replace('.jpg', '_mask.png')
        if not mask_path.exists():
            missing.append(img_path)
    
    if not missing:
        return
    
    print(f"  Generating {len(missing)} masks with SCHP...")
    try:
        from app.preprocessing.human_parsing import HumanParser
        parser = HumanParser(device="cuda", load_atr=False)
        
        for i, img_path in enumerate(missing):
            img = Image.open(img_path).convert("RGB")
            mask = parser.get_cloth_agnostic_mask(img, cloth_type="upper")
            mask_out = mask_dir / img_path.name.replace('.jpg', '_mask.png')
            mask.save(str(mask_out))
            
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(missing)}] masks generated")
        
        del parser
        torch.cuda.empty_cache()
        print(f"  {len(missing)} masks generated")
        
    except Exception as e:
        print(f"  [WARN] SCHP mask generation failed: {e}")
        print("  You can generate masks later with: python scripts/prepare_dataset.py masks --dataset <dir>")


def create_saree_mock_dataset(output_dir: str, num_samples: int = 200):
    """
    Create a mock saree dataset by augmenting available images.
    
    For real saree training data, you'll need to collect images manually:
    1. Photograph sarees flat-lay (front view, white background)
    2. Photograph models wearing sarees (front-facing, full body)
    3. Use this script to generate masks and pairs
    
    This mock dataset uses CatVTON examples + augmentation for testing
    the training pipeline.
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    out = Path(output_dir)
    img_dir = out / "train" / "image"
    cloth_dir = out / "train" / "cloth"
    mask_dir = out / "train" / "agnostic-mask"
    
    for d in [img_dir, cloth_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print(f"  Creating Mock Saree Dataset ({num_samples} samples)")
    print("=" * 50)
    
    # Source: CatVTON example images
    example_dir = CATVTON_DIR / "resource" / "demo" / "example"
    person_sources = list((example_dir / "person").rglob("*.png"))
    cloth_sources = list((example_dir / "condition").rglob("*.jpg"))
    
    if not person_sources or not cloth_sources:
        print("  [ERROR] No example images found in CatVTON/resource/demo/example/")
        return
    
    print(f"  Source: {len(person_sources)} person, {len(cloth_sources)} garment images")
    print(f"  Augmenting to {num_samples} pairs...")
    
    rng = np.random.default_rng(42)
    pairs = []
    
    for i in range(num_samples):
        # Pick random source images
        person_src = str(person_sources[i % len(person_sources)])
        cloth_src = str(cloth_sources[i % len(cloth_sources)])
        
        person_img = cv2.imread(person_src)
        cloth_img = cv2.imread(cloth_src)
        
        if person_img is None or cloth_img is None:
            continue
        
        # Augment
        h, w = person_img.shape[:2]
        
        # Random brightness
        alpha = 0.85 + rng.random() * 0.3
        beta = rng.integers(-10, 11)
        person_aug = cv2.convertScaleAbs(person_img, alpha=alpha, beta=int(beta))
        
        # Random flip
        if rng.random() > 0.5:
            person_aug = cv2.flip(person_aug, 1)
        
        # Random color jitter on cloth
        hsv = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(-15, 16)) % 180  # Hue shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.8 + rng.random() * 0.4), 0, 255)  # Sat
        cloth_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Save
        person_name = f"{i:06d}.jpg"
        cloth_name = f"{i:06d}.jpg"
        
        cv2.imwrite(str(img_dir / person_name), person_aug)
        cv2.imwrite(str(cloth_dir / cloth_name), cloth_aug)
        pairs.append(f"{person_name} {cloth_name}")
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{num_samples}] generated")
    
    # Save pairs
    pairs_file = out / "train_pairs_unpaired.txt"
    pairs_file.write_text("\n".join(pairs))
    
    # Generate masks
    print("\n  Generating masks...")
    _generate_missing_masks(out / "train", num_samples)
    
    # Create data collection guide
    guide_path = out / "SAREE_DATA_GUIDE.md"
    guide_path.write_text("""# Saree Dataset Collection Guide

## For Real Training Data

### Person Images (200-500 needed)
- Full-body front-facing photos
- Clean background (white/gray preferred)
- Good lighting, no strong shadows
- Various body types and poses
- Resolution: 768x1024 or higher

### Garment Images (50-200 needed)
- Saree flat-lay photos (front view)
- White/neutral background
- Show full drape including pallu
- Various colors and patterns
- Also include: blouse, petticoat separately if possible

### Tips
- More variety in body types = better generalization
- Consistent lighting across all shots
- Remove any text/watermarks
- Save as JPG, high quality

### After Collection
1. Place person images in `train/image/`
2. Place garment images in `train/cloth/`
3. Run: `python scripts/prepare_dataset.py masks --dataset data/saree/train --cloth-type overall`
4. Run: `python scripts/prepare_dataset.py pairs --dataset data/saree/train`
""")
    
    print(f"\n  Mock dataset created: {out}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  See {guide_path.name} for real data collection instructions")


def verify_dataset(dataset_path: str):
    """Verify a downloaded dataset is in correct CatVTON format."""
    ds = Path(dataset_path)
    
    print("=" * 50)
    print(f"  Dataset Verification: {ds}")
    print("=" * 50)
    
    issues = []
    stats = {}
    
    # Check for VITON-HD structure
    for split in ['train', 'test']:
        split_dir = ds / split
        if not split_dir.exists():
            continue
        
        for subdir in ['image', 'cloth', 'agnostic-mask']:
            d = split_dir / subdir
            if d.exists():
                count = len(list(d.glob("*.*")))
                stats[f"{split}/{subdir}"] = count
            else:
                issues.append(f"Missing: {split}/{subdir}/")
    
    # Check pairs files
    for pairs_file in ds.glob("*pairs*.txt"):
        lines = pairs_file.read_text().strip().split("\n")
        stats[f"pairs ({pairs_file.name})"] = len(lines)
    
    # Print report
    print("\n  Files:")
    for key, val in stats.items():
        print(f"    {key}: {val}")
    
    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for issue in issues:
            print(f"    [!] {issue}")
    else:
        print("\n  [OK] Dataset structure valid")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download datasets for CatVTON training")
    subparsers = parser.add_subparsers(dest="command")
    
    # vitonhd
    vh = subparsers.add_parser("vitonhd", help="Download VITON-HD dataset")
    vh.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "vitonhd"))
    vh.add_argument("--split", type=str, default="train", choices=["train", "test"])
    vh.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs")
    
    # saree mock
    sm = subparsers.add_parser("saree-mock", help="Create mock saree dataset from augmentation")
    sm.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "saree"))
    sm.add_argument("--num", type=int, default=200, help="Number of samples to generate")
    
    # verify
    vf = subparsers.add_parser("verify", help="Verify dataset structure")
    vf.add_argument("--path", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "vitonhd":
        download_vitonhd(args.output, args.split, args.max_pairs)
    elif args.command == "saree-mock":
        create_saree_mock_dataset(args.output, args.num)
    elif args.command == "verify":
        verify_dataset(args.path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
