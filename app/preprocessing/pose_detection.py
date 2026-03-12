"""
Pose Detection Module — provides body keypoints for garment alignment.

On Windows, DensePose requires detectron2 C++ binary (Linux only).
This module provides:
  1. DensePoseWrapper — wraps CatVTON's DensePose (only works on Linux/with detectron2)
  2. MediaPipePose — lightweight fallback using Google's MediaPipe (works everywhere)

MediaPipe detects 33 body landmarks including shoulders, hips, elbows, wrists, knees.
This is sufficient for garment alignment in most cases.

Usage:
    # Preferred: MediaPipe (works on Windows)
    pose = MediaPipePose()
    keypoints = pose.detect(person_image)

    # Or: DensePose (Linux only)
    pose = DensePoseWrapper(model_dir="path/to/DensePose")
    dense_map = pose.detect(person_image)
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"

# ── DensePose body part mapping ─────────────────────────────────────────────

DENSEPOSE_PARTS = {
    0: 'background',
    1: 'torso',  2: 'torso',
    3: 'right hand', 4: 'left hand',
    5: 'left foot', 6: 'right foot',
    7: 'right thigh (upper)', 8: 'left thigh (upper)',
    9: 'right thigh (lower)', 10: 'left thigh (lower)',
    11: 'right leg', 12: 'left leg',
    13: 'left big arm', 14: 'right big arm',
    15: 'left forearm', 16: 'right forearm',
    17: 'left face', 18: 'right face',
    19: 'left thigh (back)', 20: 'right thigh (back)',
    21: 'right leg (back)', 22: 'left leg (back)',
    23: 'left big arm (back)', 24: 'right big arm (back)',
}

# Grouped for mask generation
DENSE_PART_GROUPS = {
    'torso': [1, 2],
    'big arms': [13, 14, 23, 24],
    'forearms': [15, 16],
    'thighs': [7, 8, 9, 10, 19, 20],
    'legs': [11, 12, 21, 22],
    'hands': [3, 4],
    'feet': [5, 6],
    'face': [17, 18],
}


class DensePoseWrapper:
    """
    Wraps CatVTON's DensePose for body surface segmentation.
    REQUIRES detectron2 (Linux/CUDA only — will fail on Windows).
    """

    def __init__(self, model_dir: str = None, device: str = "cuda"):
        if model_dir is None:
            model_dir = str(PROJECT_ROOT / "models" / "CatVTON" / "DensePose")

        self.model_dir = model_dir
        self.device = device
        self._model = None

    def _load(self):
        """Lazy-load DensePose model."""
        if self._model is not None:
            return

        sys.path.insert(0, str(CATVTON_DIR))
        try:
            from model.DensePose import DensePose
            self._model = DensePose(model_path=self.model_dir, device=self.device)
            print("  DensePose loaded (detectron2 backend)")
        except Exception as e:
            raise RuntimeError(
                f"DensePose failed to load (requires detectron2 compiled for your OS): {e}\n"
                f"Use MediaPipePose as a fallback on Windows."
            )

    @torch.no_grad()
    def detect(self, image: Union[str, Image.Image]) -> Image.Image:
        """Returns dense body surface segmentation (grayscale, part IDs as pixel values)."""
        self._load()
        return self._model(image)

    def get_part_mask(self, dense_map: Image.Image, part_group: str) -> np.ndarray:
        """Extract binary mask for a body part group from DensePose output."""
        dense_array = np.array(dense_map)
        mask = np.zeros_like(dense_array, dtype=np.uint8)
        if part_group in DENSE_PART_GROUPS:
            for part_id in DENSE_PART_GROUPS[part_group]:
                mask[dense_array == part_id] = 255
        return mask


class MediaPipePose:
    """
    Lightweight pose detection using Google MediaPipe.
    Works on all platforms (Windows, Linux, Mac).
    Detects 33 body landmarks at ~30fps on CPU.
    """

    # Key landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
    }

    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            )
            self._available = True
        except ImportError:
            print("  [WARN] mediapipe not installed. Install with: pip install mediapipe")
            self._available = False

    def detect(self, image: Union[str, Image.Image]) -> Optional[Dict]:
        """
        Detect pose keypoints.

        Returns dict with:
            'landmarks': list of (x, y, z, visibility) for 33 points
            'keypoints': dict mapping landmark names to (x_px, y_px) pixel coords
            'shoulder_width': pixel distance between shoulders
            'torso_height': pixel distance from shoulders to hips
        """
        if not self._available:
            return None

        if isinstance(image, str):
            img = cv2.imread(image)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            raise TypeError("Input must be file path or PIL Image")

        h, w = img_rgb.shape[:2]
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z, lm.visibility))

        # Convert to pixel coordinates
        keypoints = {}
        for name, idx in self.LANDMARKS.items():
            lm = landmarks[idx]
            keypoints[name] = (int(lm[0] * w), int(lm[1] * h))

        # Compute useful measurements
        ls = keypoints['left_shoulder']
        rs = keypoints['right_shoulder']
        lh = keypoints['left_hip']
        rh = keypoints['right_hip']

        shoulder_width = np.sqrt((ls[0] - rs[0])**2 + (ls[1] - rs[1])**2)
        mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        torso_height = np.sqrt((mid_shoulder[0] - mid_hip[0])**2 +
                               (mid_shoulder[1] - mid_hip[1])**2)

        return {
            'landmarks': landmarks,
            'keypoints': keypoints,
            'shoulder_width': shoulder_width,
            'torso_height': torso_height,
            'image_size': (w, h),
        }

    def draw_pose(self, image: Union[str, Image.Image], pose_data: Dict = None) -> Image.Image:
        """Draw pose keypoints on image for visualization."""
        if isinstance(image, str):
            img = cv2.imread(image)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.array(image.convert("RGB")).copy()

        if pose_data is None:
            pose_data = self.detect(image)

        if pose_data is None:
            return Image.fromarray(img_rgb)

        keypoints = pose_data['keypoints']

        # Draw skeleton connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ]

        for start, end in connections:
            if start in keypoints and end in keypoints:
                cv2.line(img_rgb, keypoints[start], keypoints[end], (0, 255, 0), 2)

        # Draw keypoints
        for name, (x, y) in keypoints.items():
            cv2.circle(img_rgb, (x, y), 4, (255, 0, 0), -1)
            cv2.putText(img_rgb, name.split('_')[-1], (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return Image.fromarray(img_rgb)


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--backend", default="mediapipe", choices=["mediapipe", "densepose"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")

    if args.backend == "mediapipe":
        pose = MediaPipePose()
        result = pose.detect(img)
        if result:
            print(f"Detected {len(result['keypoints'])} keypoints")
            print(f"Shoulder width: {result['shoulder_width']:.0f}px")
            print(f"Torso height: {result['torso_height']:.0f}px")
            vis = pose.draw_pose(img, result)
            out = args.output or args.image.replace(".", "_pose.")
            vis.save(out)
            print(f"Pose visualization saved: {out}")
        else:
            print("No pose detected!")
    else:
        dp = DensePoseWrapper()
        dense = dp.detect(img)
        out = args.output or args.image.replace(".", "_dense.")
        dense.save(out)
        print(f"DensePose saved: {out}")
