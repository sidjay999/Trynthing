"""
Automated Saree Image Scraper — Downloads paired saree images from Bing.

Downloads two sets:
  1. "woman wearing saree full body" → person/ (model images)
  2. "saree product image flat lay" → garment/ (product images)

Then pairs them for CatVTON training.

Usage:
    # Install dependency first:
    pip install bing-image-downloader Pillow

    # Download 250 of each (creates ~250 pairs):
    python scripts/scrape_saree_auto.py --count 250 --output data/saree_raw/

    # Then organize:
    python scripts/scrape_saree.py organize --input data/saree_raw/ --output data/saree/
"""
import argparse
import os
import shutil
from pathlib import Path

from PIL import Image


def download_images(query: str, output_dir: str, count: int):
    """Download images from Bing Image Search."""
    from bing_image_downloader import downloader

    # bing_image_downloader creates a subfolder with the query name
    downloader.download(
        query,
        limit=count,
        output_dir=output_dir,
        adult_filter_off=False,
        force_replace=False,
        timeout=30,
    )
    return os.path.join(output_dir, query)


def filter_valid_images(folder: str, min_width: int = 300, min_height: int = 400):
    """Remove images that are too small or corrupt."""
    folder = Path(folder)
    removed = 0
    kept = 0

    for img_path in list(folder.glob("*")):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            img_path.unlink()
            removed += 1
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
            if w < min_width or h < min_height:
                img_path.unlink()
                removed += 1
            else:
                kept += 1
        except Exception:
            img_path.unlink()
            removed += 1

    print(f"    Kept: {kept}, Removed: {removed} (too small/corrupt)")
    return kept


def organize_into_pairs(person_folder: str, garment_folder: str, output_dir: str):
    """Move downloaded images into person/ and garment/ with matching names."""
    out = Path(output_dir)
    person_out = out / "person"
    garment_out = out / "garment"
    person_out.mkdir(parents=True, exist_ok=True)
    garment_out.mkdir(parents=True, exist_ok=True)

    person_imgs = sorted(Path(person_folder).glob("*"))
    person_imgs = [p for p in person_imgs if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]

    garment_imgs = sorted(Path(garment_folder).glob("*"))
    garment_imgs = [g for g in garment_imgs if g.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]

    # Pair by index (shuffle would be random pairing)
    n_pairs = min(len(person_imgs), len(garment_imgs))
    pairs = []

    for i in range(n_pairs):
        p_name = f"saree_{i+1:04d}.jpg"
        g_name = f"saree_{i+1:04d}.jpg"

        # Convert to JPG and save
        try:
            p_img = Image.open(person_imgs[i]).convert("RGB")
            g_img = Image.open(garment_imgs[i]).convert("RGB")

            p_img.save(str(person_out / p_name), "JPEG", quality=95)
            g_img.save(str(garment_out / g_name), "JPEG", quality=95)
            pairs.append(f"{p_name}\t{g_name}")
        except Exception as e:
            print(f"    [SKIP] Pair {i}: {e}")

    # Save pairs.txt
    (out / "pairs.txt").write_text("\n".join(pairs))
    print(f"  Created {len(pairs)} pairs in {out}")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Auto Saree Scraper")
    parser.add_argument("--count", type=int, default=250, help="Images per category")
    parser.add_argument("--output", type=str, default="data/saree_raw/")
    args = parser.parse_args()

    tmp_dir = "/tmp/saree_scrape"

    # Person images: women wearing sarees
    person_queries = [
        "indian woman wearing saree full body front",
        "model wearing silk saree standing",
        "woman in saree full length photo",
    ]

    # Garment images: saree product shots
    garment_queries = [
        "saree product image white background",
        "silk saree flat lay product photo",
        "designer saree product shot ecommerce",
    ]

    per_query = args.count // len(person_queries) + 1

    print("=" * 50)
    print("  Saree Auto Scraper")
    print("=" * 50)

    # Download person images
    print(f"\n[1/4] Downloading person images ({args.count})...")
    person_folders = []
    for q in person_queries:
        print(f"  Query: '{q}' ({per_query} images)")
        folder = download_images(q, tmp_dir, per_query)
        person_folders.append(folder)

    # Download garment images
    print(f"\n[2/4] Downloading garment images ({args.count})...")
    garment_folders = []
    for q in garment_queries:
        print(f"  Query: '{q}' ({per_query} images)")
        folder = download_images(q, tmp_dir, per_query)
        garment_folders.append(folder)

    # Merge all person images into one folder
    print("\n[3/4] Filtering & merging...")
    merged_person = Path(tmp_dir) / "_person_all"
    merged_garment = Path(tmp_dir) / "_garment_all"
    merged_person.mkdir(exist_ok=True)
    merged_garment.mkdir(exist_ok=True)

    idx = 0
    for folder in person_folders:
        if Path(folder).exists():
            for f in Path(folder).glob("*"):
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    shutil.copy2(str(f), str(merged_person / f"p_{idx:04d}{f.suffix}"))
                    idx += 1

    idx = 0
    for folder in garment_folders:
        if Path(folder).exists():
            for f in Path(folder).glob("*"):
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    shutil.copy2(str(f), str(merged_garment / f"g_{idx:04d}{f.suffix}"))
                    idx += 1

    print("  Person images:")
    filter_valid_images(str(merged_person))
    print("  Garment images:")
    filter_valid_images(str(merged_garment))

    # Organize into pairs
    print("\n[4/4] Creating pairs...")
    n = organize_into_pairs(str(merged_person), str(merged_garment), args.output)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 50}")
    print(f"  Done! {n} saree pairs saved to {args.output}")
    print(f"  Next: organize → masks → train")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
