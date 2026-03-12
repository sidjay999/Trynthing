"""
Project-wide configuration for the Virtual Try-On pipeline.
Uses RTX 4050 6GB optimized defaults.
"""
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class InferenceConfig:
    device: str = "cuda"
    precision: str = "fp16"
    width: int = 384
    height: int = 512
    num_steps: int = 30
    guidance_scale: float = 2.5
    seed: int = 42
    batch_size: int = 1
    xformers: bool = True
    cpu_offload: bool = False
    vae_slicing: bool = True
    vae_tiling: bool = True
    # CatVTON specific
    model_id: str = "zhengchong/CatVTON"
    base_model: str = "runwayml/stable-diffusion-inpainting"
    mask_type: str = "upper"  # upper | lower | overall
    repaint: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str = None) -> "InferenceConfig":
        if yaml_path is None:
            yaml_path = CONFIGS_DIR / "rtx4050.yaml"
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        flat = {}
        flat["device"] = data.get("device", "cuda")
        flat["precision"] = data.get("precision", "fp16")
        if "resolution" in data:
            flat["width"] = data["resolution"].get("width", 384)
            flat["height"] = data["resolution"].get("height", 512)
        if "inference" in data:
            flat["num_steps"] = data["inference"].get("num_steps", 30)
            flat["guidance_scale"] = data["inference"].get("guidance_scale", 2.5)
            flat["seed"] = data["inference"].get("seed", 42)
            flat["batch_size"] = data["inference"].get("batch_size", 1)
        if "optimizations" in data:
            flat["xformers"] = data["optimizations"].get("xformers", True)
            flat["cpu_offload"] = data["optimizations"].get("cpu_offload", False)
            flat["vae_slicing"] = data["optimizations"].get("vae_slicing", True)
            flat["vae_tiling"] = data["optimizations"].get("vae_tiling", True)
        if "catvton" in data:
            flat["model_id"] = data["catvton"].get("model_id", "zhengchong/CatVTON")
            flat["base_model"] = data["catvton"].get("base_model", "runwayml/stable-diffusion-inpainting")
            flat["mask_type"] = data["catvton"].get("mask_type", "upper")
            flat["repaint"] = data["catvton"].get("repaint", True)
        return cls(**flat)


# Singleton config
_config = None

def get_config(yaml_path: str = None) -> InferenceConfig:
    global _config
    if _config is None:
        _config = InferenceConfig.from_yaml(yaml_path)
    return _config
