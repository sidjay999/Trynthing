"""
CatVTON Local Inference Script — Optimized for RTX 4050 (6GB VRAM)

Usage:
    conda activate catvton
    python scripts/run_tryon.py --person data/person/photo.jpg --garment data/garment/shirt.jpg
    python scripts/run_tryon.py --person data/person/photo.jpg --garment data/garment/shirt.jpg --mask data/mask/mask.png
    python scripts/run_tryon.py --demo  # Run with bundled example images

The script auto-downloads CatVTON weights from HuggingFace on first run (~2GB).
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

# Add CatVTON to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
sys.path.insert(0, str(CATVTON_DIR))

import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def parse_args():
    parser = argparse.ArgumentParser(description="CatVTON Virtual Try-On Inference")
    parser.add_argument("--person", type=str, help="Path to person image")
    parser.add_argument("--garment", type=str, help="Path to garment image")
    parser.add_argument("--mask", type=str, default=None, help="Path to cloth-agnostic mask (white=replace region). Auto-generated if not provided.")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: data/output/result_<timestamp>.png)")
    parser.add_argument("--cloth_type", type=str, default="upper", choices=["upper", "lower", "overall"], help="Garment type for auto-masking")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    parser.add_argument("--guidance", type=float, default=2.5, help="CFG guidance scale (default: 2.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--width", type=int, default=384, help="Output width (default: 384 for 6GB VRAM)")
    parser.add_argument("--height", type=int, default=512, help="Output height (default: 512 for 6GB VRAM)")
    parser.add_argument("--demo", action="store_true", help="Run with bundled example images")
    parser.add_argument("--no-repaint", action="store_true", help="Skip repainting unmasked region")
    parser.add_argument("--use-schp", action="store_true", default=True, help="Use SCHP human parsing for mask (default: True)")
    parser.add_argument("--no-schp", action="store_true", help="Disable SCHP, use simple geometric mask")
    return parser.parse_args()


def create_simple_upper_mask(person_image: Image.Image) -> Image.Image:
    """
    Create a simple upper-body mask when AutoMasker is unavailable.
    Covers the torso region (roughly upper 60% of image, center 70% width).
    """
    w, h = person_image.size
    mask = Image.new("L", (w, h), 0)  # black = keep
    pixels = mask.load()

    # Upper body region: from ~15% to ~65% height, center ~70% width
    top = int(h * 0.15)
    bottom = int(h * 0.65)
    left = int(w * 0.15)
    right = int(w * 0.85)

    for y in range(top, bottom):
        for x in range(left, right):
            pixels[x, y] = 255  # white = replace

    # Blur the mask edges for smooth blending
    mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
    return mask


def create_overall_mask(person_image: Image.Image) -> Image.Image:
    """Create a full-body mask covering torso + legs."""
    w, h = person_image.size
    mask = Image.new("L", (w, h), 0)
    pixels = mask.load()

    top = int(h * 0.12)
    bottom = int(h * 0.90)
    left = int(w * 0.12)
    right = int(w * 0.88)

    for y in range(top, bottom):
        for x in range(left, right):
            pixels[x, y] = 255

    mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
    return mask


def create_lower_mask(person_image: Image.Image) -> Image.Image:
    """Create a lower-body mask covering legs region."""
    w, h = person_image.size
    mask = Image.new("L", (w, h), 0)
    pixels = mask.load()

    top = int(h * 0.45)
    bottom = int(h * 0.92)
    left = int(w * 0.15)
    right = int(w * 0.85)

    for y in range(top, bottom):
        for x in range(left, right):
            pixels[x, y] = 255

    mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
    return mask


def _geometric_mask(person_image: Image.Image, cloth_type: str) -> Image.Image:
    """Dispatch to simple geometric mask based on cloth type."""
    if cloth_type == "upper":
        return create_simple_upper_mask(person_image)
    elif cloth_type == "overall":
        return create_overall_mask(person_image)
    else:
        return create_lower_mask(person_image)


def repaint(person: Image.Image, mask: Image.Image, result: Image.Image) -> Image.Image:
    """Repaint: keep original background, paste try-on result only in masked region."""
    mask_np = np.array(mask.convert("L"))
    # Binary threshold
    mask_binary = (mask_np > 128).astype(np.float32)
    # Feather the edges
    from scipy.ndimage import gaussian_filter
    mask_smooth = gaussian_filter(mask_binary, sigma=3)

    person_np = np.array(person).astype(np.float32)
    result_np = np.array(result.resize(person.size)).astype(np.float32)

    # Blend: result in masked area, person in unmasked area
    blended = result_np * mask_smooth[..., None] + person_np * (1 - mask_smooth[..., None])
    return Image.fromarray(blended.astype(np.uint8))


def load_pipeline(width, height):
    """Load CatVTON pipeline with memory optimizations for 6GB VRAM."""
    local_model_dir = str(PROJECT_ROOT / "models" / "CatVTON")

    print("[1/3] Downloading CatVTON weights (cached after first run)...")
    repo_path = snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir=local_model_dir,
    )
    print(f"      Weights at: {repo_path}")

    print("[2/3] Loading pipeline (fp16 for 6GB VRAM)...")
    t0 = time.time()
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16"),
        use_tf32=True,
        device="cuda",
        skip_safety_check=True,  # Skip NSFW check for speed
    )
    print(f"      Pipeline loaded in {time.time()-t0:.1f}s")

    # Memory report
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"      VRAM: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

    return pipeline, repo_path


def run_tryon(pipeline, person_path, garment_path, mask_path, args):
    """Run a single try-on inference."""
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False,
        do_binarize=True, do_convert_grayscale=True
    )

    # Load images
    person_img = Image.open(person_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")
    print(f"  Person:  {person_img.size}  ->  ({args.width}, {args.height})")
    print(f"  Garment: {garment_img.size}  ->  ({args.width}, {args.height})")

    # Resize
    person_resized = resize_and_crop(person_img, (args.width, args.height))
    garment_resized = resize_and_padding(garment_img, (args.width, args.height))

    # Mask
    if mask_path and os.path.exists(mask_path):
        print(f"  Mask:    {mask_path} (user-provided)")
        mask = Image.open(mask_path).convert("L")
        mask = resize_and_crop(mask, (args.width, args.height))
    elif not getattr(args, 'no_schp', False):
        # Try SCHP-based parsing for accurate mask
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from app.preprocessing.human_parsing import HumanParser
            print(f"  Mask:    SCHP human parsing ({args.cloth_type})")
            hp = HumanParser(device="cuda", load_atr=False)  # LIP only to save VRAM
            mask = hp.get_cloth_agnostic_mask(person_resized, cloth_type=args.cloth_type)
            del hp
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [WARN] SCHP failed ({e}), using geometric mask")
            mask = _geometric_mask(person_resized, args.cloth_type)
    else:
        print(f"  Mask:    geometric ({args.cloth_type} body)")
        mask = _geometric_mask(person_resized, args.cloth_type)

    # Save mask for debugging
    mask_debug_path = PROJECT_ROOT / "data" / "output" / "last_mask.png"
    mask_debug_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(str(mask_debug_path))

    # Blur mask edges
    mask = mask_processor.blur(mask, blur_factor=9)

    # Generator
    generator = None
    if args.seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Clear GPU cache before inference
    gc.collect()
    torch.cuda.empty_cache()

    # Run inference
    print(f"\n[3/3] Running inference ({args.steps} steps, cfg={args.guidance}, seed={args.seed})...")
    t0 = time.time()
    result_image = pipeline(
        image=person_resized,
        condition_image=garment_resized,
        mask=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=generator,
    )[0]
    inference_time = time.time() - t0

    # Repaint original background
    if not args.no_repaint:
        result_image = repaint(person_resized, mask, result_image)

    # VRAM report
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Done in {inference_time:.1f}s | Peak VRAM: {peak:.0f}MB")

    return result_image, person_resized, garment_resized, mask


def main():
    args = parse_args()

    # Demo mode: use bundled examples
    if args.demo:
        example_dir = CATVTON_DIR / "resource" / "demo" / "example"
        args.person = str(example_dir / "person" / "men" / "model_5.png")
        args.garment = str(example_dir / "condition" / "upper" / "21514384_52353349_1000.jpg")
        print("=== DEMO MODE: using bundled example images ===")

    if not args.person or not args.garment:
        print("ERROR: --person and --garment are required (or use --demo)")
        sys.exit(1)

    if not os.path.exists(args.person):
        print(f"ERROR: Person image not found: {args.person}")
        sys.exit(1)
    if not os.path.exists(args.garment):
        print(f"ERROR: Garment image not found: {args.garment}")
        sys.exit(1)

    # Load pipeline
    pipeline, repo_path = load_pipeline(args.width, args.height)

    # Run
    result, person, garment, mask = run_tryon(
        pipeline, args.person, args.garment, args.mask, args
    )

    # Save output
    output_dir = PROJECT_ROOT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"result_{timestamp}.png"

    result.save(str(output_path))
    print(f"\n  Result saved: {output_path}")

    # Also save comparison grid
    grid_w = args.width * 3 + 10
    grid = Image.new("RGB", (grid_w, args.height), (30, 30, 30))
    grid.paste(person, (0, 0))
    grid.paste(garment, (args.width + 5, 0))
    grid.paste(result, (args.width * 2 + 10, 0))
    grid_path = output_path.parent / f"grid_{output_path.stem}.png"
    grid.save(str(grid_path))
    print(f"  Grid saved:   {grid_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
