"""
Garment Segmentation Module — removes background from garment images.

Uses rembg (U2-Net based) for automatic background removal, producing
clean garment cutouts for the try-on pipeline.

Usage:
    seg = GarmentSegmenter()
    clean_garment = seg.remove_background(garment_image)
    garment_mask = seg.get_mask(garment_image)
"""
import os
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class GarmentSegmenter:
    """Remove garment background using rembg or simple thresholding."""

    def __init__(self, use_rembg: bool = True):
        self._rembg_session = None
        self._use_rembg = use_rembg

        if use_rembg:
            try:
                import rembg
                self._rembg = rembg
                print("  GarmentSegmenter: rembg available")
            except ImportError:
                print("  [WARN] rembg not installed, using fallback. Install: pip install rembg")
                self._use_rembg = False

    def remove_background(
        self,
        image: Union[str, Image.Image],
        bg_color: tuple = (127, 127, 127),
    ) -> Image.Image:
        """
        Remove garment background and paste on neutral gray.

        Args:
            image:    Garment image (file path or PIL Image)
            bg_color: Background color (default gray — color-neutral)

        Returns:
            RGB image with garment on neutral background
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGBA")
        elif image.mode != "RGBA":
            image = image.convert("RGBA")

        if self._use_rembg:
            # rembg produces RGBA with transparent background
            result = self._rembg.remove(image)
        else:
            # Simple fallback: assume white/near-white background
            result = self._simple_bg_remove(image)

        # Composite on neutral background
        bg = Image.new("RGB", result.size, bg_color)
        if result.mode == "RGBA":
            bg.paste(result, mask=result.split()[3])
        else:
            bg.paste(result)
        return bg

    def get_mask(self, image: Union[str, Image.Image]) -> Image.Image:
        """Get binary garment mask (white=garment, black=background)."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGBA")
        elif image.mode != "RGBA":
            image = image.convert("RGBA")

        if self._use_rembg:
            result = self._rembg.remove(image)
            alpha = np.array(result.split()[3])
            mask = (alpha > 128).astype(np.uint8) * 255
        else:
            mask = self._simple_mask(image)

        return Image.fromarray(mask).convert("L")

    def _simple_bg_remove(self, image: Image.Image) -> Image.Image:
        """Fallback: remove near-white background by thresholding."""
        img_array = np.array(image.convert("RGB"))
        # Detect white-ish pixels
        white_mask = np.all(img_array > 240, axis=2)
        # Create alpha channel
        alpha = np.ones(img_array.shape[:2], dtype=np.uint8) * 255
        alpha[white_mask] = 0
        result = Image.fromarray(
            np.concatenate([img_array, alpha[..., None]], axis=2), "RGBA"
        )
        return result

    def _simple_mask(self, image: Image.Image) -> np.ndarray:
        """Fallback: get garment mask via thresholding."""
        img_array = np.array(image.convert("RGB"))
        white_mask = np.all(img_array > 240, axis=2)
        mask = (~white_mask).astype(np.uint8) * 255
        return mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seg = GarmentSegmenter()
    img = Image.open(args.image)
    result = seg.remove_background(img)
    out = args.output or args.image.replace(".", "_clean.")
    result.save(out)
    print(f"Clean garment saved: {out}")
