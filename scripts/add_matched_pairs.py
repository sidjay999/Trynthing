"""
Add matched saree pairs to the organized training dataset.

Takes the manually collected matched pairs (1.jpg/1.png ... 30.jpg/30.png)
from saree_raw/ and adds them to data/saree/train/ with proper naming.

Usage (WSL):
    cd /mnt/c/Users/JAY/OneDrive/Desktop/trynthing
    PYTHONPATH=CatVTON:. ~/miniconda3/bin/python scripts/add_matched_pairs.py
"""
import os
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_PERSON = PROJECT_ROOT / "data" / "saree_raw" / "person"
RAW_GARMENT = PROJECT_ROOT / "data" / "saree_raw" / "garment"
TRAIN_DIR = PROJECT_ROOT / "data" / "saree" / "train"
TARGET_SIZE = (768, 1024)  # width, height


def resize_and_save(img_path: Path, out_path: Path):
    """Load image, resize to 768x1024, save as JPG."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    img.save(str(out_path), "JPEG", quality=95)


def main():
    person_dir = TRAIN_DIR / "person"
    garment_dir = TRAIN_DIR / "garment"
    pairs_file = TRAIN_DIR / "pairs.txt"

    # Find the next available index
    existing = [f.stem for f in person_dir.glob("saree_*_person.jpg")]
    if existing:
        max_idx = max(int(f.split("_")[1]) for f in existing)
    else:
        max_idx = -1

    print(f"Existing pairs: {len(existing)} (max index: {max_idx})")

    # Find matched pairs in raw (files named 1.jpg, 2.jpg ... 30.jpg for person, 1.png ... 30.png for garment)
    matched_person = {}
    for f in RAW_PERSON.glob("*"):
        stem = f.stem
        if stem.isdigit():
            matched_person[int(stem)] = f

    matched_garment = {}
    for f in RAW_GARMENT.glob("*"):
        stem = f.stem
        if stem.isdigit():
            matched_garment[int(stem)] = f

    # Find common pairs
    common = sorted(set(matched_person.keys()) & set(matched_garment.keys()))
    print(f"Found {len(common)} matched pairs to add: {common}")

    if not common:
        print("No matched pairs found! Expected files named 1.jpg, 2.jpg... in saree_raw/person/ and 1.png, 2.png... in saree_raw/garment/")
        return

    # Add pairs
    new_pairs = []
    for i, num in enumerate(common):
        new_idx = max_idx + 1 + i
        person_name = f"saree_{new_idx:04d}_person.jpg"
        garment_name = f"saree_{new_idx:04d}_garment.jpg"

        print(f"  [{i+1}/{len(common)}] {num}.jpg + {num}.png -> {person_name} + {garment_name}")

        resize_and_save(matched_person[num], person_dir / person_name)
        resize_and_save(matched_garment[num], garment_dir / garment_name)
        new_pairs.append(f"{person_name}\t{garment_name}")

    # Append to pairs.txt
    with open(pairs_file, "a") as f:
        for pair in new_pairs:
            f.write(pair + "\n")

    total = len(existing) + len(new_pairs)
    print(f"\nDone! Added {len(new_pairs)} matched pairs.")
    print(f"Total pairs now: {total}")
    print(f"pairs.txt updated: {pairs_file}")
    print(f"\nNext: re-zip and transfer to remote GPU for retraining")


if __name__ == "__main__":
    main()
