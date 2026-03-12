"""
AI Virtual Try-On — HuggingFace Spaces App
Uses ZeroGPU (@spaces.GPU) for free shared GPU access.
"""
import os
import sys
import gc

# Add CatVTON to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CatVTON"))
sys.path.insert(0, os.path.dirname(__file__))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# ZeroGPU: only import 'spaces' on HuggingFace (not available locally)
try:
    import spaces
    IS_HF_SPACES = True
except ImportError:
    IS_HF_SPACES = False

# Global model references (loaded once)
pipeline = None
mask_processor = None
automasker = None
schp_lip = None
schp_atr = None
repo_path = None
_models_loaded = False


def load_models():
    """Load all models (called once on first inference)."""
    global pipeline, mask_processor, automasker, schp_lip, schp_atr, repo_path, _models_loaded

    if _models_loaded:
        return

    from huggingface_hub import snapshot_download
    from diffusers.image_processor import VaeImageProcessor
    from model.pipeline import CatVTONPipeline
    from utils import init_weight_dtype

    print("[Loading] Downloading CatVTON model...")
    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

    print("[Loading] Initializing pipeline (bf16)...")
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("bf16"),
        use_tf32=True,
        device="cuda",
        skip_safety_check=True,
    )

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False,
        do_binarize=True, do_convert_grayscale=True
    )

    # Try full AutoMasker (DensePose + SCHP)
    try:
        from model.cloth_masker import AutoMasker
        automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device="cuda",
        )
        print("[Loading] Full AutoMasker loaded (DensePose + SCHP)")
    except Exception as e:
        print(f"[Loading] AutoMasker failed: {e}, using SCHP fallback")
        automasker = None
        try:
            from model.SCHP import SCHP
            schp_dir = os.path.join(repo_path, "SCHP")
            schp_lip = SCHP(
                ckpt_path=os.path.join(schp_dir, "exp-schp-201908261155-lip.pth"),
                device="cuda"
            )
            schp_atr = SCHP(
                ckpt_path=os.path.join(schp_dir, "exp-schp-201908301523-atr.pth"),
                device="cuda"
            )
            print("[Loading] SCHP masking loaded (LIP + ATR)")
        except Exception as e2:
            print(f"[Loading] SCHP also failed: {e2}")

    _models_loaded = True
    print("[Loading] Ready!")


def generate_mask(person_image, cloth_type):
    """Generate cloth-agnostic mask."""
    if automasker is not None:
        result = automasker(person_image, cloth_type)
        return result['mask']

    if schp_lip is not None and schp_atr is not None:
        from model.cloth_masker import AutoMasker as AM
        lip_result = schp_lip(person_image)
        atr_result = schp_atr(person_image)
        w, h = person_image.size
        dummy_dp = Image.new("L", (w, h), 0)
        return AM.cloth_agnostic_mask(dummy_dp, lip_result, atr_result, part=cloth_type)

    # Geometric fallback (if all models fail)
    w, h = person_image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if cloth_type == "upper":
        draw.rectangle([int(w * 0.15), int(h * 0.18), int(w * 0.85), int(h * 0.58)], fill=255)
    elif cloth_type == "lower":
        draw.rectangle([int(w * 0.18), int(h * 0.45), int(w * 0.82), int(h * 0.88)], fill=255)
    else:
        draw.rectangle([int(w * 0.15), int(h * 0.18), int(w * 0.85), int(h * 0.85)], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 40))


def _run_inference(person_img, garment_img, cloth_type, seed):
    """Core try-on inference logic."""
    if person_img is None or garment_img is None:
        raise gr.Error("Please upload both a photo and a garment image.")

    load_models()

    from utils import resize_and_crop, resize_and_padding, repaint_result

    person_image = Image.fromarray(person_img).convert("RGB")
    garment_image = Image.fromarray(garment_img).convert("RGB")

    width, height = 768, 1024
    person_resized = resize_and_crop(person_image, (width, height))
    garment_resized = resize_and_padding(garment_image, (width, height))

    mask = generate_mask(person_resized, cloth_type)
    mask = mask_processor.blur(mask, blur_factor=9)

    generator = None
    seed_int = int(seed)
    if seed_int >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed_int)

    result = pipeline(
        image=person_resized,
        condition_image=garment_resized,
        mask=mask,
        num_inference_steps=50,
        guidance_scale=2.5,
        height=height,
        width=width,
        generator=generator,
    )[0]

    result = repaint_result(result, person_resized, mask)

    gc.collect()
    torch.cuda.empty_cache()

    return np.array(result)


# Apply @spaces.GPU decorator only on HuggingFace Spaces
if IS_HF_SPACES:
    @spaces.GPU(duration=120)
    def try_on(person_img, garment_img, cloth_type, seed):
        """Run try-on with ZeroGPU allocation."""
        return _run_inference(person_img, garment_img, cloth_type, seed)
else:
    def try_on(person_img, garment_img, cloth_type, seed):
        """Run try-on locally (no ZeroGPU)."""
        return _run_inference(person_img, garment_img, cloth_type, seed)


# ─── Gradio Interface ───

with gr.Blocks(
    title="AI Virtual Try-On",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
    ),
    css="""
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .subtitle { text-align: center; color: #666; margin-bottom: 2em; }
    footer { display: none !important; }
    """
) as demo:

    gr.HTML("<h1 class='main-title'>👔 AI Virtual Try-On</h1>")
    gr.HTML("<p class='subtitle'>Upload a photo of yourself and a garment — see yourself wearing it in seconds</p>")

    with gr.Row():
        with gr.Column(scale=1):
            person_input = gr.Image(label="Your Photo", type="numpy")
            garment_input = gr.Image(label="Garment Image", type="numpy")

            with gr.Row():
                cloth_type = gr.Dropdown(
                    choices=["upper", "lower", "overall"],
                    value="upper",
                    label="Garment Type"
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)

            run_btn = gr.Button("✨ Generate Try-On", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Try-On Result", type="numpy")

    gr.HTML("""
    <div style="text-align: center; margin-top: 2em; color: #888; font-size: 0.9em;">
        <p>🔒 <strong>Privacy:</strong> Your photos are processed in memory and never stored.</p>
        <p>⚡ DensePose + SCHP masking • 768×1024 resolution • 50-step inference</p>
    </div>
    """)

    run_btn.click(
        fn=try_on,
        inputs=[person_input, garment_input, cloth_type, seed_input],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
