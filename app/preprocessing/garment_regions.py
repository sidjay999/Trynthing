"""
Saree & Complex Garment Region Splitter

Sarees and other draped garments need special handling because they wrap
around the body in multiple zones:
  - Underskirt / petticoat (lower body)
  - Pleats / front drape (center torso)
  - Pallu / shoulder drape (over shoulder, across chest)
  - Blouse region (upper torso, arms)

This module splits garment images into these semantic regions for
zone-specific TPS warping before feeding to CatVTON.

Usage:
    splitter = GarmentRegionSplitter()
    regions = splitter.split_saree(garment_image)
    # regions = {'blouse': Image, 'pleats': Image, 'pallu': Image, 'underskirt': Image}
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# ── Garment type definitions ───────────────────────────────────────────────

GARMENT_TYPES = {
    'tshirt': {'regions': ['body'], 'complexity': 'simple'},
    'shirt': {'regions': ['body', 'collar', 'sleeves'], 'complexity': 'simple'},
    'dress': {'regions': ['upper', 'lower'], 'complexity': 'moderate'},
    'suit': {'regions': ['jacket', 'pants'], 'complexity': 'moderate'},
    'saree': {'regions': ['blouse', 'pleats', 'pallu', 'underskirt'], 'complexity': 'complex'},
    'lehenga': {'regions': ['blouse', 'skirt', 'dupatta'], 'complexity': 'complex'},
    'kurta': {'regions': ['body', 'sleeves'], 'complexity': 'simple'},
}

# Body region proportions for splitting
REGION_PROPORTIONS = {
    'saree': {
        'blouse': (0.0, 0.35, 0.15, 0.85),      # top 35%, center 70%
        'pallu': (0.0, 0.50, 0.55, 1.0),         # top 50%, right side
        'pleats': (0.30, 0.80, 0.20, 0.60),      # mid section, center-left
        'underskirt': (0.50, 1.0, 0.15, 0.85),   # bottom 50%
    },
    'lehenga': {
        'blouse': (0.0, 0.35, 0.15, 0.85),
        'skirt': (0.35, 1.0, 0.10, 0.90),
        'dupatta': (0.0, 0.60, 0.50, 1.0),
    },
    'suit': {
        'jacket': (0.0, 0.55, 0.05, 0.95),
        'pants': (0.45, 1.0, 0.15, 0.85),
    },
    'dress': {
        'upper': (0.0, 0.45, 0.10, 0.90),
        'lower': (0.40, 1.0, 0.10, 0.90),
    },
}


class GarmentRegionSplitter:
    """Split garment images into semantic regions for zone-specific processing."""

    def __init__(self):
        pass

    def detect_garment_type(self, image: Union[str, Image.Image]) -> str:
        """
        Detect garment type from image.
        For now uses aspect ratio heuristic. Future: use classifier.

        Returns: one of GARMENT_TYPES keys
        """
        if isinstance(image, str):
            image = Image.open(image)
        w, h = image.size
        ratio = h / w

        # Heuristic: tall images = likely full-body garments
        if ratio > 2.0:
            return 'saree'  # Very tall = draped garment
        elif ratio > 1.5:
            return 'dress'
        else:
            return 'tshirt'

    def split(
        self,
        image: Union[str, Image.Image],
        garment_type: str = None,
    ) -> Dict[str, Image.Image]:
        """
        Split garment into semantic regions.

        Args:
            image:        Garment image
            garment_type: One of GARMENT_TYPES keys, or auto-detected

        Returns:
            Dict mapping region name to cropped Image
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if garment_type is None:
            garment_type = self.detect_garment_type(image)

        if garment_type not in REGION_PROPORTIONS:
            # Simple garment — return as-is
            return {'body': image}

        w, h = image.size
        regions = {}
        proportions = REGION_PROPORTIONS[garment_type]

        for region_name, (top_frac, bot_frac, left_frac, right_frac) in proportions.items():
            top = int(h * top_frac)
            bottom = int(h * bot_frac)
            left = int(w * left_frac)
            right = int(w * right_frac)

            # Ensure minimum crop size
            if (right - left) < 20 or (bottom - top) < 20:
                continue

            region = image.crop((left, top, right, bottom))
            regions[region_name] = region

        return regions

    def split_saree(self, image: Union[str, Image.Image]) -> Dict[str, Image.Image]:
        """Convenience: split specifically as saree."""
        return self.split(image, garment_type='saree')

    def split_with_masks(
        self,
        image: Union[str, Image.Image],
        garment_type: str = None,
    ) -> Dict[str, Dict]:
        """
        Split garment with corresponding masks for each region.
        Returns dict with {'image': Image, 'mask': Image, 'bbox': tuple} per region.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if garment_type is None:
            garment_type = self.detect_garment_type(image)

        if garment_type not in REGION_PROPORTIONS:
            w, h = image.size
            mask = Image.new("L", (w, h), 255)
            return {'body': {'image': image, 'mask': mask, 'bbox': (0, 0, w, h)}}

        w, h = image.size
        regions = {}
        proportions = REGION_PROPORTIONS[garment_type]

        for region_name, (top_frac, bot_frac, left_frac, right_frac) in proportions.items():
            top = int(h * top_frac)
            bottom = int(h * bot_frac)
            left = int(w * left_frac)
            right = int(w * right_frac)

            if (right - left) < 20 or (bottom - top) < 20:
                continue

            region_img = image.crop((left, top, right, bottom))
            region_mask = Image.new("L", (right - left, bottom - top), 255)

            # Feather mask edges
            region_mask = region_mask.filter(ImageFilter.GaussianBlur(radius=10))

            regions[region_name] = {
                'image': region_img,
                'mask': region_mask,
                'bbox': (left, top, right, bottom),
            }

        return regions

    def reassemble(
        self,
        regions: Dict[str, Image.Image],
        original_size: Tuple[int, int],
        garment_type: str,
    ) -> Image.Image:
        """
        Reassemble processed regions back into a full garment image.

        Args:
            regions:       Dict mapping region name to processed Image
            original_size: (width, height) of original garment
            garment_type:  Type used for splitting
        """
        w, h = original_size
        result = Image.new("RGB", (w, h), (127, 127, 127))

        if garment_type not in REGION_PROPORTIONS:
            if 'body' in regions:
                return regions['body'].resize((w, h))
            return result

        proportions = REGION_PROPORTIONS[garment_type]

        for region_name, (top_frac, bot_frac, left_frac, right_frac) in proportions.items():
            if region_name not in regions:
                continue

            top = int(h * top_frac)
            bottom = int(h * bot_frac)
            left = int(w * left_frac)
            right = int(w * right_frac)

            region = regions[region_name]
            region_resized = region.resize((right - left, bottom - top), Image.LANCZOS)
            result.paste(region_resized, (left, top))

        return result

    def visualize_regions(
        self,
        image: Union[str, Image.Image],
        garment_type: str = None,
    ) -> Image.Image:
        """Draw region boundaries on image for visualization."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if garment_type is None:
            garment_type = self.detect_garment_type(image)

        vis = np.array(image.copy())
        w, h = image.size

        if garment_type not in REGION_PROPORTIONS:
            return image

        colors = {
            'blouse': (255, 0, 0),      # Red
            'pleats': (0, 255, 0),       # Green
            'pallu': (0, 0, 255),        # Blue
            'underskirt': (255, 255, 0), # Yellow
            'skirt': (0, 255, 0),
            'dupatta': (0, 0, 255),
            'jacket': (255, 0, 0),
            'pants': (0, 255, 0),
            'upper': (255, 0, 0),
            'lower': (0, 255, 0),
        }

        proportions = REGION_PROPORTIONS[garment_type]
        for region_name, (top_frac, bot_frac, left_frac, right_frac) in proportions.items():
            top = int(h * top_frac)
            bottom = int(h * bot_frac)
            left = int(w * left_frac)
            right = int(w * right_frac)

            color = colors.get(region_name, (128, 128, 128))
            cv2.rectangle(vis, (left, top), (right, bottom), color, 2)
            cv2.putText(vis, region_name, (left + 5, top + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return Image.fromarray(vis)


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--type", type=str, default=None, choices=list(GARMENT_TYPES.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    splitter = GarmentRegionSplitter()
    img = Image.open(args.image).convert("RGB")
    garment_type = args.type or splitter.detect_garment_type(img)
    print(f"Garment type: {garment_type}")

    regions = splitter.split(img, garment_type)
    out_dir = args.output_dir or os.path.dirname(args.image)

    for name, region in regions.items():
        out_path = os.path.join(out_dir, f"region_{name}.png")
        region.save(out_path)
        print(f"  {name}: {region.size} -> {out_path}")

    vis = splitter.visualize_regions(img, garment_type)
    vis.save(os.path.join(out_dir, "regions_vis.png"))
    print(f"Visualization saved")
