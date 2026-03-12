"""
Human Parsing Module — wraps CatVTON's SCHP model for body part segmentation.

SCHP (Self-Correction Human Parsing) outputs per-pixel body part labels:
  LIP dataset (20 classes): Background, Hat, Hair, Glove, Sunglasses, Upper-clothes,
    Dress, Coat, Socks, Pants, Jumpsuits, Scarf, Skirt, Face, Left-arm, Right-arm,
    Left-leg, Right-leg, Left-shoe, Right-shoe

  ATR dataset (18 classes): Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt,
    Pants, Dress, Belt, Left-shoe, Right-shoe, Face, Left-leg, Right-leg,
    Left-arm, Right-arm, Bag, Scarf

Usage:
    parser = HumanParser(model_dir="path/to/SCHP/checkpoints")
    result = parser.parse(person_image)
    mask = parser.get_cloth_agnostic_mask(person_image, cloth_type="upper")
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Add CatVTON to path for SCHP imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
sys.path.insert(0, str(CATVTON_DIR))

from model.SCHP import SCHP, dataset_settings

# ── Label mappings ──────────────────────────────────────────────────────────

LIP_LABELS = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 'Sunglasses': 4,
    'Upper-clothes': 5, 'Dress': 6, 'Coat': 7, 'Socks': 8, 'Pants': 9,
    'Jumpsuits': 10, 'Scarf': 11, 'Skirt': 12, 'Face': 13, 'Left-arm': 14,
    'Right-arm': 15, 'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18,
    'Right-shoe': 19
}

ATR_LABELS = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 'Upper-clothes': 4,
    'Skirt': 5, 'Pants': 6, 'Dress': 7, 'Belt': 8, 'Left-shoe': 9,
    'Right-shoe': 10, 'Face': 11, 'Left-leg': 12, 'Right-leg': 13,
    'Left-arm': 14, 'Right-arm': 15, 'Bag': 16, 'Scarf': 17
}

# Parts to mask out for each garment type (white in cloth-agnostic mask)
MASK_PARTS_LIP = {
    'upper': ['Upper-clothes', 'Coat', 'Dress'],
    'lower': ['Pants', 'Skirt', 'Dress'],
    'overall': ['Upper-clothes', 'Coat', 'Dress', 'Pants', 'Skirt', 'Jumpsuits'],
}

# Parts to always preserve (never mask)
PRESERVE_PARTS_LIP = [
    'Background', 'Hat', 'Hair', 'Sunglasses', 'Face',
    'Left-shoe', 'Right-shoe', 'Glove', 'Scarf'
]


class HumanParser:
    """Wrapper around CatVTON's SCHP model for human body part parsing."""

    def __init__(
        self,
        model_dir: str = None,
        device: str = "cuda",
        load_lip: bool = True,
        load_atr: bool = True,
    ):
        if model_dir is None:
            model_dir = str(PROJECT_ROOT / "models" / "CatVTON" / "SCHP")

        self.device = device
        self.model_dir = model_dir
        self.schp_lip = None
        self.schp_atr = None

        lip_ckpt = os.path.join(model_dir, "exp-schp-201908261155-lip.pth")
        atr_ckpt = os.path.join(model_dir, "exp-schp-201908301523-atr.pth")

        if load_lip and os.path.exists(lip_ckpt):
            print(f"  Loading SCHP-LIP from {lip_ckpt}...")
            self.schp_lip = SCHP(lip_ckpt, device)
            print(f"  SCHP-LIP loaded (20 classes)")

        if load_atr and os.path.exists(atr_ckpt):
            print(f"  Loading SCHP-ATR from {atr_ckpt}...")
            self.schp_atr = SCHP(atr_ckpt, device)
            print(f"  SCHP-ATR loaded (18 classes)")

    @torch.no_grad()
    def parse_lip(self, image: Union[str, Image.Image]) -> Image.Image:
        """Run LIP human parsing. Returns palette-indexed segmentation."""
        assert self.schp_lip is not None, "SCHP-LIP model not loaded"
        return self.schp_lip(image)

    @torch.no_grad()
    def parse_atr(self, image: Union[str, Image.Image]) -> Image.Image:
        """Run ATR human parsing. Returns palette-indexed segmentation."""
        assert self.schp_atr is not None, "SCHP-ATR model not loaded"
        return self.schp_atr(image)

    @torch.no_grad()
    def parse(self, image: Union[str, Image.Image]) -> Dict[str, Image.Image]:
        """Run both LIP and ATR parsing. Returns dict of results."""
        results = {}
        if self.schp_lip is not None:
            results['lip'] = self.schp_lip(image)
        if self.schp_atr is not None:
            results['atr'] = self.schp_atr(image)
        return results

    def get_part_mask(
        self,
        parsing: Image.Image,
        part_names: List[str],
        dataset: str = 'lip'
    ) -> np.ndarray:
        """
        Extract binary mask for specific body parts from a parsing result.

        Args:
            parsing:    Palette-indexed segmentation from SCHP
            part_names: List of part names (e.g. ['Upper-clothes', 'Coat'])
            dataset:    'lip' or 'atr'

        Returns:
            Binary mask (H, W) with 255 for selected parts
        """
        labels = LIP_LABELS if dataset == 'lip' else ATR_LABELS
        parse_array = np.array(parsing)
        mask = np.zeros_like(parse_array, dtype=np.uint8)
        for part in part_names:
            if part in labels:
                mask[parse_array == labels[part]] = 255
        return mask

    def get_cloth_agnostic_mask(
        self,
        image: Union[str, Image.Image],
        cloth_type: str = "upper",
        blur_radius: int = 5,
    ) -> Image.Image:
        """
        Generate a cloth-agnostic mask for CatVTON input.

        White (255) = region to replace with new garment
        Black (0)   = region to preserve

        Args:
            image:       Person image
            cloth_type:  'upper', 'lower', or 'overall'
            blur_radius: Gaussian blur for smooth mask edges

        Returns:
            Grayscale mask image
        """
        # Run LIP parsing
        lip_result = self.parse_lip(image)
        parse_array = np.array(lip_result)

        # Start with blank mask (all black = preserve everything)
        h, w = parse_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Mark clothing regions to replace
        parts_to_mask = MASK_PARTS_LIP.get(cloth_type, MASK_PARTS_LIP['upper'])
        for part_name in parts_to_mask:
            if part_name in LIP_LABELS:
                mask[parse_array == LIP_LABELS[part_name]] = 255

        # Also include arm regions for upper/overall (arms get new sleeves)
        if cloth_type in ('upper', 'overall'):
            for arm in ['Left-arm', 'Right-arm']:
                mask[parse_array == LIP_LABELS[arm]] = 255

        # Include leg regions for lower/overall
        if cloth_type in ('lower', 'overall'):
            for leg in ['Left-leg', 'Right-leg']:
                mask[parse_array == LIP_LABELS[leg]] = 255

        # Morphological cleanup: close small holes, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Optional blur for smooth edges
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)

        return Image.fromarray(mask).convert("L")

    def visualize_parsing(
        self,
        parsing: Image.Image,
        original: Image.Image = None,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlay parsing result on original image for visualization."""
        parse_vis = parsing.convert("RGB")
        if original is not None:
            original_rgb = original.convert("RGB").resize(parse_vis.size)
            parse_vis = Image.blend(original_rgb, parse_vis, alpha)
        return parse_vis


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--type", type=str, default="upper", choices=["upper", "lower", "overall"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    hp = HumanParser()
    img = Image.open(args.image).convert("RGB")

    print("Running human parsing...")
    mask = hp.get_cloth_agnostic_mask(img, cloth_type=args.type)

    out = args.output or args.image.replace(".", f"_mask_{args.type}.")
    mask.save(out)
    print(f"Mask saved: {out}")
