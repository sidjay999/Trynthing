"""
Saree Dataset Scraper — Downloads paired saree images from e-commerce.

Scrapes saree product pages to get:
  1. Garment image (flat-lay / product shot) → garment/
  2. Person image (model wearing saree) → person/
  3. Creates pairs.txt mapping

Sources:
  - Amazon.in saree listings
  - Open fashion datasets on HuggingFace

Usage:
    # Scrape from Amazon.in search results
    python scripts/scrape_saree.py amazon --pages 10 --output data/saree_raw/

    # Download from HuggingFace fashion datasets
    python scripts/scrape_saree.py hf --output data/saree_raw/ --limit 500

    # Download from open image URLs list
    python scripts/scrape_saree.py urls --file saree_urls.txt --output data/saree_raw/

    # Organize scraped images into train/val format
    python scripts/scrape_saree.py organize --input data/saree_raw/ --output data/saree/
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


# ═══════════════════════════════════════════════════════════════════════════
#  Amazon Scraper
# ═══════════════════════════════════════════════════════════════════════════

def scrape_amazon(output_dir: str, pages: int = 5, keyword: str = "saree women"):
    """
    Scrape saree images from Amazon.in search results.
    Downloads product images (multiple views per product).
    """
    out = Path(output_dir)
    garment_dir = out / "garment"
    person_dir = out / "person"
    garment_dir.mkdir(parents=True, exist_ok=True)
    person_dir.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("Install required packages: pip install requests beautifulsoup4")
        return

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-IN,en;q=0.9',
    }

    all_pairs = []
    img_count = 0

    for page in range(1, pages + 1):
        query = keyword.replace(" ", "+")
        url = f"https://www.amazon.in/s?k={query}&page={page}"
        print(f"  [Page {page}/{pages}] Fetching {url}")

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find product image containers
            img_tags = soup.select('img.s-image')

            for img_tag in img_tags:
                src = img_tag.get('src', '')
                if not src or 'sprite' in src:
                    continue

                # Get high-res version
                srcset = img_tag.get('srcset', '')
                if srcset:
                    # Parse srcset for highest resolution
                    parts = srcset.split(',')
                    best_url = parts[-1].strip().split(' ')[0]
                else:
                    best_url = src

                # Download image
                try:
                    img_hash = hashlib.md5(best_url.encode()).hexdigest()[:8]
                    img_name = f"saree_{img_count:04d}_{img_hash}.jpg"

                    # Classify: product shots (white bg) → garment, model shots → person
                    img_path = garment_dir / img_name
                    urllib.request.urlretrieve(best_url, str(img_path))

                    # Check if it's a person or garment image
                    pil_img = Image.open(img_path)
                    w, h = pil_img.size
                    if w >= 200 and h >= 200:
                        # Heuristic: tall images with models tend to be person images
                        if h / w > 1.3:
                            final_path = person_dir / img_name
                            img_path.rename(final_path)
                        img_count += 1
                    else:
                        img_path.unlink()  # Too small

                except Exception as e:
                    pass

            time.sleep(2)  # Be polite to servers

        except Exception as e:
            print(f"    Error on page {page}: {e}")
            continue

    print(f"  Downloaded {img_count} images")
    return img_count


# ═══════════════════════════════════════════════════════════════════════════
#  HuggingFace Fashion Dataset
# ═══════════════════════════════════════════════════════════════════════════

def download_hf_fashion(output_dir: str, limit: int = 500):
    """
    Download saree/dress images from HuggingFace fashion datasets.
    Uses open datasets that have garment + person pairs.
    """
    out = Path(output_dir)
    garment_dir = out / "garment"
    person_dir = out / "person"
    garment_dir.mkdir(parents=True, exist_ok=True)
    person_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return

    # Try DressCode dataset (has paired images)
    print("  Loading DressCode dataset from HuggingFace...")
    try:
        ds = load_dataset(
            "mattmdjaga/dresscode_1024",
            split="train",
            streaming=True,
        )

        count = 0
        pairs = []
        for item in ds:
            if count >= limit:
                break

            try:
                # DressCode format: image (person), cloth (garment)
                person_img = item.get('image')
                cloth_img = item.get('cloth')

                if person_img and cloth_img:
                    p_name = f"person_{count:04d}.jpg"
                    g_name = f"garment_{count:04d}.jpg"

                    if isinstance(person_img, Image.Image):
                        person_img.convert("RGB").save(str(person_dir / p_name), quality=95)
                    if isinstance(cloth_img, Image.Image):
                        cloth_img.convert("RGB").save(str(garment_dir / g_name), quality=95)

                    pairs.append(f"{p_name}\t{g_name}")
                    count += 1

                    if count % 50 == 0:
                        print(f"    Downloaded {count}/{limit} pairs")

            except Exception as e:
                continue

        # Save pairs
        pairs_path = out / "pairs.txt"
        pairs_path.write_text("\n".join(pairs))
        print(f"  Downloaded {count} pairs → {out}")

    except Exception as e:
        print(f"  DressCode download failed: {e}")
        print("  Trying alternative datasets...")

        # Fallback: try other fashion datasets
        try:
            ds = load_dataset("yisol/IDM-VTON", split="train", streaming=True)
            count = 0
            for item in ds:
                if count >= limit:
                    break
                try:
                    person_img = item.get('image')
                    cloth_img = item.get('cloth')
                    if person_img and cloth_img:
                        p_name = f"person_{count:04d}.jpg"
                        g_name = f"garment_{count:04d}.jpg"
                        person_img.convert("RGB").save(str(person_dir / p_name), quality=95)
                        cloth_img.convert("RGB").save(str(garment_dir / g_name), quality=95)
                        count += 1
                except:
                    continue
            print(f"  Downloaded {count} pairs from IDM-VTON")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")


# ═══════════════════════════════════════════════════════════════════════════
#  URL List Downloader
# ═══════════════════════════════════════════════════════════════════════════

def download_from_urls(urls_file: str, output_dir: str):
    """
    Download images from a text file with URLs.
    Format: one URL per line, or TAB-separated: person_url<TAB>garment_url
    """
    out = Path(output_dir)
    garment_dir = out / "garment"
    person_dir = out / "person"
    garment_dir.mkdir(parents=True, exist_ok=True)
    person_dir.mkdir(parents=True, exist_ok=True)

    lines = Path(urls_file).read_text().strip().split("\n")
    count = 0

    for i, line in enumerate(lines):
        parts = line.strip().split("\t")

        try:
            if len(parts) == 2:
                # Paired format: person_url \t garment_url
                p_url, g_url = parts
                p_name = f"person_{i:04d}.jpg"
                g_name = f"garment_{i:04d}.jpg"
                urllib.request.urlretrieve(p_url, str(person_dir / p_name))
                urllib.request.urlretrieve(g_url, str(garment_dir / g_name))
                count += 1
            elif len(parts) == 1:
                # Single URL — auto-classify
                url = parts[0]
                img_name = f"img_{i:04d}.jpg"
                img_path = out / img_name
                urllib.request.urlretrieve(url, str(img_path))

                # Classify by aspect ratio
                img = Image.open(img_path)
                w, h = img.size
                if h / w > 1.3:
                    img_path.rename(person_dir / img_name)
                else:
                    img_path.rename(garment_dir / img_name)
                count += 1

            if count % 20 == 0:
                print(f"    Downloaded {count} images")
            time.sleep(0.5)

        except Exception as e:
            print(f"    [FAIL] Line {i}: {e}")

    print(f"  Downloaded {count} images from URL list")


# ═══════════════════════════════════════════════════════════════════════════
#  Organize into train/val
# ═══════════════════════════════════════════════════════════════════════════

def organize_scraped(input_dir: str, output_dir: str, split_ratio: float = 0.9):
    """Organize scraped images into CatVTON dataset format."""
    inp = Path(input_dir)
    out = Path(output_dir)

    person_dir = inp / "person"
    garment_dir = inp / "garment"

    if not person_dir.exists() or not garment_dir.exists():
        print("Error: input_dir must have person/ and garment/ subdirectories")
        return

    persons = sorted([f for f in person_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
    garments = sorted([f for f in garment_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])

    print(f"  Found: {len(persons)} person, {len(garments)} garment images")

    # Resize all to 768x1024
    for split in ['train', 'val']:
        for sub in ['person', 'garment', 'mask', 'parse']:
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    n_train = int(min(len(persons), len(garments)) * split_ratio)

    pairs_train = []
    pairs_val = []

    for i, (p, g) in enumerate(zip(persons, garments)):
        split = 'train' if i < n_train else 'val'

        # Resize and save
        p_name = f"saree_{i:04d}_person.jpg"
        g_name = f"saree_{i:04d}_garment.jpg"

        try:
            p_img = Image.open(p).convert("RGB")
            g_img = Image.open(g).convert("RGB")

            # Resize to 768x1024 (CatVTON default)
            p_img = p_img.resize((768, 1024), Image.LANCZOS)
            g_img = g_img.resize((768, 1024), Image.LANCZOS)

            p_img.save(str(out / split / "person" / p_name), quality=95)
            g_img.save(str(out / split / "garment" / g_name), quality=95)

            pair_line = f"{p_name}\t{g_name}"
            if split == 'train':
                pairs_train.append(pair_line)
            else:
                pairs_val.append(pair_line)

        except Exception as e:
            print(f"    [FAIL] Pair {i}: {e}")

    # Save pairs.txt
    (out / "train" / "pairs.txt").write_text("\n".join(pairs_train))
    (out / "val" / "pairs.txt").write_text("\n".join(pairs_val))

    print(f"  Organized: {len(pairs_train)} train, {len(pairs_val)} val pairs")
    print(f"  Output: {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Saree Dataset Scraper")
    subparsers = parser.add_subparsers(dest="command")

    # Amazon scraper
    amz = subparsers.add_parser("amazon", help="Scrape from Amazon.in")
    amz.add_argument("--pages", type=int, default=5)
    amz.add_argument("--output", type=str, default="data/saree_raw/")
    amz.add_argument("--keyword", type=str, default="saree women")

    # HuggingFace downloader
    hf = subparsers.add_parser("hf", help="Download from HuggingFace datasets")
    hf.add_argument("--output", type=str, default="data/saree_raw/")
    hf.add_argument("--limit", type=int, default=500)

    # URL list
    urls = subparsers.add_parser("urls", help="Download from URL list")
    urls.add_argument("--file", type=str, required=True)
    urls.add_argument("--output", type=str, default="data/saree_raw/")

    # Organize
    org = subparsers.add_parser("organize", help="Organize into train/val")
    org.add_argument("--input", type=str, required=True)
    org.add_argument("--output", type=str, required=True)
    org.add_argument("--split", type=float, default=0.9)

    args = parser.parse_args()

    if args.command == "amazon":
        scrape_amazon(args.output, args.pages, args.keyword)
    elif args.command == "hf":
        download_hf_fashion(args.output, args.limit)
    elif args.command == "urls":
        download_from_urls(args.file, args.output)
    elif args.command == "organize":
        organize_scraped(args.input, args.output, args.split)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
