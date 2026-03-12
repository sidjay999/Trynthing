"""
Dataset Organizer — structure data for CatVTON training/evaluation.

Expected dataset structure:
    data/
    ├── train/
    │   ├── person/          # Person images
    │   ├── garment/         # Garment images (flat-lay or on-model)
    │   ├── mask/            # Cloth-agnostic masks (auto-generated)
    │   ├── parse/           # SCHP parsing results (auto-generated)
    │   └── pairs.txt        # person_image <tab> garment_image
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure)

Usage:
    # Organize loose images into dataset structure
    python scripts/prepare_dataset.py organize --input raw_images/ --output data/train/

    # Generate pairs.txt from existing structure
    python scripts/prepare_dataset.py pairs --dataset data/train/

    # Batch generate masks
    python scripts/prepare_dataset.py masks --dataset data/train/
    
    # Validate dataset
    python scripts/prepare_dataset.py validate --dataset data/train/

    # Augment dataset
    python scripts/prepare_dataset.py augment --dataset data/train/ --factor 3
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
sys.path.insert(0, str(CATVTON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


# ═══════════════════════════════════════════════════════════════════════════
#  ORGANIZE: structure raw images into dataset format
# ═══════════════════════════════════════════════════════════════════════════

def organize_dataset(input_dir: str, output_dir: str, split_ratio: float = 0.9):
    """
    Organize raw images into train/val dataset structure.

    Expects input_dir to have:
      - person/ subfolder with person images
      - garment/ subfolder with garment images 
      - OR mixed images (will be sorted by aspect ratio heuristic)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output structure
    for split in ['train', 'val']:
        for subdir in ['person', 'garment', 'mask', 'parse', 'pose']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    # Find person and garment images
    person_dir = input_path / "person"
    garment_dir = input_path / "garment"

    if person_dir.exists() and garment_dir.exists():
        person_images = sorted([f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
        garment_images = sorted([f for f in garment_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
    else:
        # Auto-classify by aspect ratio: tall = person, squarish = garment
        all_images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
        person_images = []
        garment_images = []
        for f in all_images:
            img = Image.open(f)
            w, h = img.size
            if h / w > 1.3:
                person_images.append(f)
            else:
                garment_images.append(f)
        print(f"  Auto-classified: {len(person_images)} person, {len(garment_images)} garment")

    # Split
    n_train_p = int(len(person_images) * split_ratio)
    n_train_g = int(len(garment_images) * split_ratio)

    splits = {
        'train': {
            'person': person_images[:n_train_p],
            'garment': garment_images[:n_train_g],
        },
        'val': {
            'person': person_images[n_train_p:],
            'garment': garment_images[n_train_g:],
        }
    }

    for split, data in splits.items():
        for category, images in data.items():
            for img_path in images:
                dst = output_path / split / category / img_path.name
                shutil.copy2(str(img_path), str(dst))
        print(f"  {split}: {len(data['person'])} person, {len(data['garment'])} garment")

    print(f"Dataset organized at: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  PAIRS: generate pairs.txt
# ═══════════════════════════════════════════════════════════════════════════

def generate_pairs(dataset_dir: str, strategy: str = "all"):
    """
    Generate pairs.txt mapping person images to garment images.

    Strategies:
      - "all":     every person × every garment (cartesian product)
      - "matched": 1:1 matching by filename index
      - "random":  random N pairings
    """
    ds = Path(dataset_dir)
    person_dir = ds / "person"
    garment_dir = ds / "garment"

    person_images = sorted([f.name for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
    garment_images = sorted([f.name for f in garment_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])

    pairs = []
    if strategy == "all":
        for p in person_images:
            for g in garment_images:
                pairs.append(f"{p}\t{g}")
    elif strategy == "matched":
        for p, g in zip(person_images, garment_images):
            pairs.append(f"{p}\t{g}")
    elif strategy == "random":
        rng = np.random.default_rng(42)
        n = max(len(person_images), len(garment_images)) * 2
        for _ in range(n):
            p = rng.choice(person_images)
            g = rng.choice(garment_images)
            pairs.append(f"{p}\t{g}")

    pairs_path = ds / "pairs.txt"
    pairs_path.write_text("\n".join(pairs))
    print(f"  Generated {len(pairs)} pairs -> {pairs_path}")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  MASKS: batch generate cloth-agnostic masks using SCHP
# ═══════════════════════════════════════════════════════════════════════════

def generate_masks(dataset_dir: str, cloth_type: str = "upper", device: str = "cuda"):
    """Batch generate cloth-agnostic masks and SCHP parses for all person images."""
    import torch

    ds = Path(dataset_dir)
    person_dir = ds / "person"
    mask_dir = ds / "mask"
    parse_dir = ds / "parse"
    mask_dir.mkdir(parents=True, exist_ok=True)
    parse_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
    print(f"  Generating masks for {len(images)} images ({cloth_type})...")

    # Load SCHP
    from app.preprocessing.human_parsing import HumanParser
    parser = HumanParser(device=device, load_atr=False)

    stats = {'success': 0, 'failed': 0}
    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")

            # SCHP parsing
            parse_result = parser.parse_lip(img)
            parse_out = parse_dir / f"{img_path.stem}.png"
            parse_result.save(str(parse_out))

            # Cloth-agnostic mask
            mask = parser.get_cloth_agnostic_mask(img, cloth_type=cloth_type)
            mask_out = mask_dir / f"{img_path.stem}.png"
            mask.save(str(mask_out))

            stats['success'] += 1
            if (i + 1) % 10 == 0 or i == len(images) - 1:
                print(f"    [{i+1}/{len(images)}] processed")

        except Exception as e:
            print(f"    [FAIL] {img_path.name}: {e}")
            stats['failed'] += 1

    # Cleanup GPU
    del parser
    torch.cuda.empty_cache()

    print(f"  Done: {stats['success']} masks generated, {stats['failed']} failed")
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  AUGMENT: expand dataset with augmentations
# ═══════════════════════════════════════════════════════════════════════════

def augment_dataset(dataset_dir: str, factor: int = 3):
    """
    Augment person images with transformations:
      - Horizontal flip
      - Brightness / contrast jitter
      - Small rotation (-5 to +5 degrees)
    """
    ds = Path(dataset_dir)
    person_dir = ds / "person"
    images = sorted([f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])

    print(f"  Augmenting {len(images)} images x{factor}...")
    rng = np.random.default_rng(42)
    count = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        for j in range(factor):
            augmented = img.copy()
            suffix = f"_aug{j}"

            # Random horizontal flip
            if rng.random() > 0.5:
                augmented = cv2.flip(augmented, 1)
                suffix += "_flip"

            # Random brightness/contrast
            alpha = 0.8 + rng.random() * 0.4  # 0.8-1.2
            beta = rng.integers(-15, 16)       # -15 to +15
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=int(beta))

            # Random small rotation
            angle = rng.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            out_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            cv2.imwrite(str(person_dir / out_name), augmented)
            count += 1

    print(f"  Created {count} augmented images")
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATE: check dataset quality and completeness
# ═══════════════════════════════════════════════════════════════════════════

def validate_dataset(dataset_dir: str) -> Dict:
    """
    Validate dataset structure and image quality.

    Checks:
      - Directory structure is correct
      - All images are readable
      - Image dimensions are consistent
      - Masks exist for all person images
      - Pairs.txt references valid files
    """
    ds = Path(dataset_dir)
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    # Check directories
    required_dirs = ['person', 'garment']
    optional_dirs = ['mask', 'parse', 'pose']

    for d in required_dirs:
        if not (ds / d).exists():
            report['errors'].append(f"Missing required directory: {d}/")
            report['valid'] = False

    for d in optional_dirs:
        if not (ds / d).exists():
            report['warnings'].append(f"Missing optional directory: {d}/ (will be auto-generated)")

    if not report['valid']:
        return report

    # Count and validate images
    for subdir in ['person', 'garment', 'mask', 'parse']:
        dir_path = ds / subdir
        if not dir_path.exists():
            report['stats'][subdir] = 0
            continue

        images = [f for f in dir_path.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
        report['stats'][subdir] = len(images)

        # Check each image
        sizes = []
        for img_path in images:
            try:
                img = Image.open(img_path)
                img.verify()
                sizes.append(img.size)
            except Exception as e:
                report['errors'].append(f"Corrupt image: {subdir}/{img_path.name}: {e}")

        if sizes:
            unique_sizes = set(sizes)
            if len(unique_sizes) > 1:
                report['warnings'].append(
                    f"{subdir}/: {len(unique_sizes)} different image sizes found "
                    f"(min: {min(sizes)}, max: {max(sizes)})"
                )

    # Check masks coverage
    person_names = {f.stem for f in (ds / 'person').iterdir() if f.suffix.lower() in IMG_EXTENSIONS}
    if (ds / 'mask').exists():
        mask_names = {f.stem for f in (ds / 'mask').iterdir() if f.suffix.lower() in IMG_EXTENSIONS}
        missing_masks = person_names - mask_names
        if missing_masks:
            report['warnings'].append(
                f"{len(missing_masks)} person images missing masks: {list(missing_masks)[:5]}..."
            )

    # Check pairs.txt
    pairs_path = ds / "pairs.txt"
    if pairs_path.exists():
        lines = pairs_path.read_text().strip().split("\n")
        report['stats']['pairs'] = len(lines)

        invalid_pairs = 0
        for line in lines[:100]:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                invalid_pairs += 1
                continue
            p, g = parts
            if not (ds / 'person' / p).exists():
                invalid_pairs += 1
            if not (ds / 'garment' / g).exists():
                invalid_pairs += 1
        if invalid_pairs:
            report['errors'].append(f"{invalid_pairs} invalid pairs in pairs.txt")
    else:
        report['warnings'].append("No pairs.txt found (run 'pairs' command to generate)")

    # Summary
    print("\n  Dataset Validation Report")
    print("  " + "=" * 40)
    for key, val in report['stats'].items():
        print(f"  {key:>10}: {val} images")
    if report['errors']:
        print(f"\n  ERRORS ({len(report['errors'])}):")
        for e in report['errors']:
            print(f"    [!] {e}")
        report['valid'] = False
    if report['warnings']:
        print(f"\n  WARNINGS ({len(report['warnings'])}):")
        for w in report['warnings']:
            print(f"    [~] {w}")
    if report['valid']:
        print("\n  [OK] Dataset is valid!")
    else:
        print("\n  [FAIL] Dataset has errors")

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CatVTON Dataset Preparation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # organize
    org = subparsers.add_parser("organize", help="Organize raw images into dataset structure")
    org.add_argument("--input", type=str, required=True, help="Input directory with raw images")
    org.add_argument("--output", type=str, required=True, help="Output dataset directory")
    org.add_argument("--split", type=float, default=0.9, help="Train/val split ratio")

    # pairs
    pairs = subparsers.add_parser("pairs", help="Generate pairs.txt")
    pairs.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    pairs.add_argument("--strategy", type=str, default="all", choices=["all", "matched", "random"])

    # masks
    masks = subparsers.add_parser("masks", help="Batch generate cloth-agnostic masks")
    masks.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    masks.add_argument("--cloth-type", type=str, default="upper", choices=["upper", "lower", "overall"])
    masks.add_argument("--device", type=str, default="cuda")

    # augment
    aug = subparsers.add_parser("augment", help="Augment dataset with transformations")
    aug.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    aug.add_argument("--factor", type=int, default=3, help="Augmentation factor")

    # validate
    val = subparsers.add_parser("validate", help="Validate dataset structure and quality")
    val.add_argument("--dataset", type=str, required=True, help="Dataset directory")

    args = parser.parse_args()

    if args.command == "organize":
        organize_dataset(args.input, args.output, args.split)
    elif args.command == "pairs":
        generate_pairs(args.dataset, args.strategy)
    elif args.command == "masks":
        generate_masks(args.dataset, args.cloth_type, args.device)
    elif args.command == "augment":
        augment_dataset(args.dataset, args.factor)
    elif args.command == "validate":
        validate_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
