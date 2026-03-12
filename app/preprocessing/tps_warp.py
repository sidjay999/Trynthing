"""
Thin Plate Spline (TPS) Warping Module

Warps garment regions to match body shape using TPS (Thin Plate Spline)
transformation. Essential for complex garments like sarees where the
drape must follow body contours.

TPS is a non-rigid deformation that smoothly maps source points to
target points, creating natural-looking cloth deformation.

Usage:
    warper = TPSWarper()
    warped = warper.warp(garment_img, source_pts, target_pts)
    # Or auto-warp using pose keypoints:
    warped = warper.auto_warp(garment_img, person_img, cloth_type="upper")
"""
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union


class TPSWarper:
    """
    Thin Plate Spline cloth warping.

    Smoothly deforms garment images to match target body shape using
    control point correspondences between garment and body.
    """

    def __init__(self):
        pass

    def warp(
        self,
        image: Union[Image.Image, np.ndarray],
        source_points: np.ndarray,
        target_points: np.ndarray,
        output_size: Tuple[int, int] = None,
    ) -> Image.Image:
        """
        Apply TPS warp to image using control point correspondences.

        Args:
            image:         Source image (garment)
            source_points: Nx2 float array of control points on source image
            target_points: Nx2 float array of target positions
            output_size:   (width, height) of output, defaults to source size

        Returns:
            Warped image
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        h, w = img.shape[:2]
        if output_size is None:
            output_size = (w, h)

        # Reshape for OpenCV TPS
        src = source_points.reshape(1, -1, 2).astype(np.float32)
        dst = target_points.reshape(1, -1, 2).astype(np.float32)

        # Create TPS transformer
        tps = cv2.createThinPlateSplineShapeTransformer()

        # Match points 1-to-1
        n = source_points.shape[0]
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]

        # Estimate transformation
        tps.estimateTransformation(dst, src, matches)

        # Apply transformation
        result = tps.warpImage(img)

        # Resize to output size if needed
        if result.shape[1] != output_size[0] or result.shape[0] != output_size[1]:
            result = cv2.resize(result, output_size, interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(result)

    def create_grid_points(
        self,
        width: int,
        height: int,
        nx: int = 5,
        ny: int = 7,
        margin: float = 0.05,
    ) -> np.ndarray:
        """
        Create a regular grid of control points.

        Args:
            width, height: Image dimensions
            nx, ny:        Grid subdivisions in x and y
            margin:        Fraction of dimension to leave as border

        Returns:
            (nx*ny) x 2 array of grid points
        """
        x_margin = int(width * margin)
        y_margin = int(height * margin)

        xs = np.linspace(x_margin, width - x_margin, nx)
        ys = np.linspace(y_margin, height - y_margin, ny)

        points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        return points

    def auto_warp_upper(
        self,
        garment: Image.Image,
        person_keypoints: dict,
        target_size: Tuple[int, int] = None,
    ) -> Image.Image:
        """
        Auto-warp an upper garment using pose keypoints.

        Maps garment corners and midpoints to person's shoulder, hip,
        and arm keypoints for natural fit.
        """
        if target_size is None:
            target_size = garment.size

        gw, gh = garment.size
        tw, th = target_size

        kp = person_keypoints  # dict with 'left_shoulder', 'right_shoulder', etc.

        # Source points: garment control grid
        src = self.create_grid_points(gw, gh, nx=4, ny=5)

        # Target points: body-aligned grid
        ls = np.array(kp.get('left_shoulder', (tw * 0.25, th * 0.15)))
        rs = np.array(kp.get('right_shoulder', (tw * 0.75, th * 0.15)))
        lh = np.array(kp.get('left_hip', (tw * 0.30, th * 0.60)))
        rh = np.array(kp.get('right_hip', (tw * 0.70, th * 0.60)))

        # Create a body-fitted target grid
        # Interpolate between shoulder line and hip line
        dst = np.zeros_like(src)
        for i, (sx, sy) in enumerate(src):
            # Normalize source position
            fx = sx / gw
            fy = sy / gh

            # Interpolate between shoulder and hip based on y position
            top = ls * (1 - fx) + rs * fx
            bot = lh * (1 - fx) + rh * fx
            point = top * (1 - fy) + bot * fy

            dst[i] = point

        return self.warp(garment, src, dst, output_size=target_size)

    def auto_warp_saree_region(
        self,
        region_image: Image.Image,
        region_name: str,
        person_keypoints: dict,
        target_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Auto-warp a saree region based on body keypoints.

        Different regions map to different body areas:
          - blouse: shoulders to upper torso
          - pleats: torso center, flowing downward
          - pallu: across chest, over shoulder
          - underskirt: hips to ankles
        """
        tw, th = target_size
        gw, gh = region_image.size
        kp = person_keypoints

        ls = np.array(kp.get('left_shoulder', (tw * 0.25, th * 0.15)))
        rs = np.array(kp.get('right_shoulder', (tw * 0.75, th * 0.15)))
        lh = np.array(kp.get('left_hip', (tw * 0.30, th * 0.55)))
        rh = np.array(kp.get('right_hip', (tw * 0.70, th * 0.55)))
        lk = np.array(kp.get('left_knee', (tw * 0.30, th * 0.80)))
        rk = np.array(kp.get('right_knee', (tw * 0.70, th * 0.80)))

        if region_name == 'blouse':
            # Map to upper torso (shoulders to mid-torso)
            src = self.create_grid_points(gw, gh, nx=4, ny=4)
            mid_s = (ls + rs) / 2
            mid_h = (lh + rh) / 2
            dst = np.zeros_like(src)
            for i, (sx, sy) in enumerate(src):
                fx, fy = sx / gw, sy / gh
                top = ls * (1 - fx) + rs * fx
                bot = lh * (1 - fx) + rh * fx
                # Only use top ~40% of body
                point = top + (bot - top) * fy * 0.6
                dst[i] = point

        elif region_name == 'pleats':
            # Map to center-front torso flowing down
            src = self.create_grid_points(gw, gh, nx=3, ny=5)
            dst = np.zeros_like(src)
            for i, (sx, sy) in enumerate(src):
                fx, fy = sx / gw, sy / gh
                # Center-biased mapping
                cx = tw * (0.30 + fx * 0.40)
                cy = th * (0.30 + fy * 0.55)
                dst[i] = [cx, cy]

        elif region_name == 'pallu':
            # Map diagonally across body (right hip to left shoulder)
            src = self.create_grid_points(gw, gh, nx=3, ny=5)
            dst = np.zeros_like(src)
            for i, (sx, sy) in enumerate(src):
                fx, fy = sx / gw, sy / gh
                # Diagonal mapping: from lower-right to upper-left
                start = rh.copy()
                end = ls + np.array([-tw * 0.1, -th * 0.05])
                point = start + (end - start) * fy
                point[0] += (fx - 0.5) * tw * 0.25
                dst[i] = point

        elif region_name == 'underskirt':
            # Map to lower body (hips to ankles)
            src = self.create_grid_points(gw, gh, nx=4, ny=5)
            dst = np.zeros_like(src)
            for i, (sx, sy) in enumerate(src):
                fx, fy = sx / gw, sy / gh
                top = lh * (1 - fx) + rh * fx
                bot = lk * (1 - fx) + rk * fx
                point = top + (bot - top) * fy
                dst[i] = point

        else:
            # Default: simple grid warp
            src = self.create_grid_points(gw, gh, nx=4, ny=4)
            dst = self.create_grid_points(tw, th, nx=4, ny=4)

        return self.warp(region_image, src, dst, output_size=target_size)


class ComplexGarmentPipeline:
    """
    End-to-end pipeline for processing complex garments (sarees, lehengas, etc.)
    through region splitting -> TPS warping -> CatVTON inference.
    """

    def __init__(self, garment_type: str = "saree"):
        from app.preprocessing.garment_regions import GarmentRegionSplitter
        self.splitter = GarmentRegionSplitter()
        self.warper = TPSWarper()
        self.garment_type = garment_type

    def preprocess(
        self,
        garment_image: Union[str, Image.Image],
        person_keypoints: dict,
        target_size: Tuple[int, int] = (384, 512),
    ) -> Dict[str, Image.Image]:
        """
        Full preprocessing for complex garment:
        1. Split into regions
        2. TPS warp each region to body
        3. Return warped regions ready for try-on

        Returns:
            Dict of warped region images
        """
        if isinstance(garment_image, str):
            garment_image = Image.open(garment_image).convert("RGB")

        # Step 1: Split
        regions = self.splitter.split(garment_image, self.garment_type)

        # Step 2: Warp each region
        warped = {}
        for name, region_img in regions.items():
            try:
                warped_region = self.warper.auto_warp_saree_region(
                    region_img, name, person_keypoints, target_size
                )
                warped[name] = warped_region
            except Exception as e:
                print(f"  [WARN] TPS warp failed for region '{name}': {e}")
                warped[name] = region_img.resize(target_size, Image.LANCZOS)

        return warped

    def compose_garment(
        self,
        warped_regions: Dict[str, Image.Image],
        target_size: Tuple[int, int] = (384, 512),
    ) -> Image.Image:
        """
        Compose warped regions into a single garment condition image
        for CatVTON input.

        Uses alpha blending to layer regions (pallu on top of pleats, etc.)
        """
        w, h = target_size
        result = Image.new("RGB", (w, h), (127, 127, 127))

        # Layer order (back to front) for saree
        layer_order = ['underskirt', 'blouse', 'pleats', 'pallu',
                       'skirt', 'jacket', 'pants', 'dupatta',
                       'upper', 'lower', 'body']

        for layer in layer_order:
            if layer in warped_regions:
                region = warped_regions[layer].resize((w, h), Image.LANCZOS)
                # Simple paste — future: add alpha blending at edges
                result.paste(region, (0, 0))

        return result


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--garment", type=str, required=True)
    parser.add_argument("--type", type=str, default="saree", choices=list(GARMENT_TYPES.keys()))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    warper = TPSWarper()
    garment = Image.open(args.garment).convert("RGB")
    w, h = garment.size

    # Demo: simple grid warp with mild deformation
    src = warper.create_grid_points(w, h, nx=4, ny=5)
    # Add random perturbation to target points
    dst = src.copy()
    np.random.seed(42)
    dst += np.random.randn(*dst.shape).astype(np.float32) * 10

    result = warper.warp(garment, src, dst)
    out = args.output or args.garment.replace(".", "_warped.")
    result.save(out)
    print(f"Warped garment saved: {out}")
