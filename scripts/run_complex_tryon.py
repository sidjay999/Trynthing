"""
Complex Garment Try-On Script

Handles sarees, lehengas, suits, and other complex garments by:
1. Detecting body pose (MediaPipe)
2. Splitting garment into regions (blouse, pleats, pallu, etc.)
3. TPS-warping each region to match body shape
4. Composing warped regions into a single condition image
5. Running CatVTON inference

Usage:
    conda activate catvton
    python scripts/run_complex_tryon.py --person data/person/photo.jpg \
        --garment data/garment/saree.jpg --type saree
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
sys.path.insert(0, str(CATVTON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


def parse_args():
    parser = argparse.ArgumentParser(description="Complex Garment Try-On (Saree/Lehenga/Suit)")
    parser.add_argument("--person", type=str, required=True, help="Path to person image")
    parser.add_argument("--garment", type=str, required=True, help="Path to garment image")
    parser.add_argument("--type", type=str, default="saree",
                        choices=["saree", "lehenga", "suit", "dress", "tshirt", "shirt", "kurta"],
                        help="Garment type")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print(f"  Complex Garment Try-On: {args.type}")
    print("=" * 50)

    # ── 1. Pose detection ──
    print("\n[1/5] Detecting body pose...")
    from app.preprocessing.pose_detection import MediaPipePose
    pose_detector = MediaPipePose()
    person_img = Image.open(args.person).convert("RGB")
    pose_data = pose_detector.detect(person_img)

    if pose_data:
        print(f"  Found {len(pose_data['keypoints'])} keypoints")
        print(f"  Shoulder width: {pose_data['shoulder_width']:.0f}px")
        print(f"  Torso height: {pose_data['torso_height']:.0f}px")
    else:
        print("  [WARN] No pose detected, using default keypoints")
        w, h = person_img.size
        pose_data = {
            'keypoints': {
                'left_shoulder': (int(w * 0.3), int(h * 0.2)),
                'right_shoulder': (int(w * 0.7), int(h * 0.2)),
                'left_hip': (int(w * 0.35), int(h * 0.55)),
                'right_hip': (int(w * 0.65), int(h * 0.55)),
                'left_knee': (int(w * 0.35), int(h * 0.8)),
                'right_knee': (int(w * 0.65), int(h * 0.8)),
            }
        }

    # ── 2. Region splitting ──
    print("\n[2/5] Splitting garment into regions...")
    from app.preprocessing.garment_regions import GarmentRegionSplitter
    splitter = GarmentRegionSplitter()
    garment_img = Image.open(args.garment).convert("RGB")
    regions = splitter.split(garment_img, args.type)
    for name, region in regions.items():
        print(f"  {name}: {region.size}")

    # Save region visualization
    vis = splitter.visualize_regions(garment_img, args.type)
    output_dir = PROJECT_ROOT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis.save(str(output_dir / "regions_debug.png"))

    # ── 3. TPS warping ──
    print("\n[3/5] TPS warping regions to body...")
    from app.preprocessing.tps_warp import TPSWarper
    warper = TPSWarper()
    target_size = (args.width, args.height)

    warped_regions = {}
    for name, region in regions.items():
        try:
            warped = warper.auto_warp_saree_region(
                region, name, pose_data['keypoints'], target_size
            )
            warped_regions[name] = warped
            print(f"  {name}: warped to {warped.size}")
        except Exception as e:
            print(f"  [WARN] {name} warp failed: {e}, using resize")
            warped_regions[name] = region.resize(target_size, Image.LANCZOS)

    # ── 4. Compose condition image ──
    print("\n[4/5] Composing condition image...")
    from app.preprocessing.tps_warp import ComplexGarmentPipeline
    composed = Image.new("RGB", target_size, (127, 127, 127))

    # Layer order
    layer_order = ['underskirt', 'blouse', 'pleats', 'pallu',
                   'skirt', 'jacket', 'pants', 'dupatta',
                   'upper', 'lower', 'body']
    for layer in layer_order:
        if layer in warped_regions:
            composed.paste(warped_regions[layer], (0, 0))

    composed.save(str(output_dir / "composed_garment.png"))
    print(f"  Composed garment saved")

    # ── 5. Mask + CatVTON inference ──
    print("\n[5/5] Running CatVTON inference...")
    local_model_dir = str(PROJECT_ROOT / "models" / "CatVTON")

    repo_path = snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir=local_model_dir,
    )

    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16"),
        use_tf32=True,
        device="cuda",
        skip_safety_check=True,
    )

    # Prepare inputs
    person_resized = resize_and_crop(person_img, target_size)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False,
        do_binarize=True, do_convert_grayscale=True
    )

    # Use SCHP mask for overall garment
    cloth_type = "overall" if args.type in ("saree", "lehenga", "dress") else "upper"
    try:
        from app.preprocessing.human_parsing import HumanParser
        hp = HumanParser(device="cuda", load_atr=False)
        mask = hp.get_cloth_agnostic_mask(person_resized, cloth_type=cloth_type)
        del hp
    except Exception:
        # Fallback geometric mask
        w, h = person_resized.size
        mask = Image.new("L", (w, h), 0)
        pixels = mask.load()
        for y in range(int(h * 0.12), int(h * 0.90)):
            for x in range(int(w * 0.12), int(w * 0.88)):
                pixels[x, y] = 255
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=15))

    mask = mask_processor.blur(mask, blur_factor=9)

    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    t0 = time.time()
    result_image = pipeline(
        image=person_resized,
        condition_image=composed,
        mask=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=generator,
    )[0]
    inference_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Done in {inference_time:.1f}s | Peak VRAM: {peak_vram:.0f}MB")

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"complex_{args.type}_{timestamp}.png"

    result_image.save(str(output_path))
    print(f"\n  Result: {output_path}")

    # Grid: person | garment | composed | result
    grid_w = args.width * 4 + 15
    grid = Image.new("RGB", (grid_w, args.height), (30, 30, 30))
    grid.paste(person_resized, (0, 0))
    grid.paste(garment_img.resize(target_size, Image.LANCZOS), (args.width + 5, 0))
    grid.paste(composed, (args.width * 2 + 10, 0))
    grid.paste(result_image, (args.width * 3 + 15, 0))
    grid_path = output_path.parent / f"grid_{output_path.stem}.png"
    grid.save(str(grid_path))
    print(f"  Grid:   {grid_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
