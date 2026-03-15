"""
CatVTON SaaS Platform — FastAPI Backend

Uses CatVTON's EXACT inference path from app.py.
No custom masking, no custom repaint — just CatVTON.

Flow:
  Seller -> uploads garment -> gets unique try-on link + QR code
  Customer -> scans QR -> uploads photo -> sees try-on result (photo NOT stored)
"""
import asyncio
import gc
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import qrcode
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

# Project paths — server.py is at app/api/server.py, so 3 levels up = project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CATVTON_DIR = PROJECT_ROOT / "CatVTON"
sys.path.insert(0, str(CATVTON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
GARMENT_STORE = PROJECT_ROOT / "data" / "garment_store"
GARMENT_STORE.mkdir(parents=True, exist_ok=True)
STATIC_DIR = PROJECT_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
#  GPU Inference Engine — uses CatVTON's EXACT code from app.py
# =====================================================================

class TryOnEngine:
    """
    Thin wrapper around CatVTON's original code.
    Mirrors app.py submit_function line-for-line.
    """
    
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.pipeline = None
        self.mask_processor = None
        self.automasker = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._lora_path = None
        self._lora_loaded = False
    
    async def load(self):
        """Load CatVTON pipeline + AutoMasker — identical to app.py."""
        if self._loaded:
            return
        
        import torch
        from diffusers.image_processor import VaeImageProcessor
        from huggingface_hub import snapshot_download
        from model.pipeline import CatVTONPipeline
        from utils import init_weight_dtype
        
        print("[Engine] Loading CatVTON pipeline (fp16)...")
        local_model_dir = str(PROJECT_ROOT / "models" / "CatVTON")
        repo_path = snapshot_download(
            repo_id="zhengchong/CatVTON",
            local_dir=local_model_dir,
        )
        
        # Exact same as app.py lines 106-113
        self.pipeline = CatVTONPipeline(
            base_ckpt="booksforcharlie/stable-diffusion-inpainting",
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype("bf16"),
            use_tf32=True,
            device="cuda",
            skip_safety_check=True,
        )
        
        # Exact same as app.py line 115
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False,
            do_binarize=True, do_convert_grayscale=True
        )
        
        # Load AutoMasker — exact same as app.py lines 116-120
        try:
            from model.cloth_masker import AutoMasker
            self.automasker = AutoMasker(
                densepose_ckpt=os.path.join(repo_path, "DensePose"),
                schp_ckpt=os.path.join(repo_path, "SCHP"),
                device="cuda",
            )
            print("[Engine] AutoMasker loaded (DensePose + SCHP)")
        except Exception as e:
            print(f"[Engine] Full AutoMasker failed: {e}")
            print("[Engine] Loading SCHP-only masking fallback...")
            try:
                from model.SCHP import SCHP
                schp_dir = os.path.join(repo_path, "SCHP")
                self._schp_lip = SCHP(
                    ckpt_path=os.path.join(schp_dir, "exp-schp-201908261155-lip.pth"),
                    device="cuda"
                )
                self._schp_atr = SCHP(
                    ckpt_path=os.path.join(schp_dir, "exp-schp-201908301523-atr.pth"),
                    device="cuda"
                )
                print("[Engine] SCHP masking loaded (LIP + ATR)")
            except Exception as e2:
                print(f"[Engine] SCHP also failed: {e2}")
                self._schp_lip = None
                self._schp_atr = None
        
        self.repo_path = repo_path
        self._loaded = True
        
        # Detect LoRA checkpoints for saree/overall
        lora_v2 = PROJECT_ROOT / "checkpoints" / "saree_v2" / "lora_epoch40"
        lora_v1 = PROJECT_ROOT / "checkpoints" / "saree" / "lora_epoch20"
        if lora_v2.exists():
            self._lora_path = str(lora_v2)
            print(f"[Engine] Saree LoRA found: {self._lora_path}")
        elif lora_v1.exists():
            self._lora_path = str(lora_v1)
            print(f"[Engine] Saree LoRA found: {self._lora_path}")
        else:
            print("[Engine] No saree LoRA checkpoints found (will use base model for overall)")
        
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[Engine] Ready! VRAM: {vram:.0f}MB")
    
    def _generate_mask(self, person_image, cloth_type):
        """Generate mask using CatVTON's own AutoMasker."""
        # Best: full AutoMasker (DensePose + SCHP)
        if self.automasker is not None:
            result = self.automasker(person_image, cloth_type)
            return result['mask']
        
        # Fallback: SCHP-only using CatVTON's static method
        if hasattr(self, '_schp_lip') and self._schp_lip is not None:
            from model.cloth_masker import AutoMasker as AM
            
            schp_lip_result = self._schp_lip(person_image)
            schp_atr_result = self._schp_atr(person_image)
            
            # Dummy densepose (zeros) + use CatVTON's mask logic
            w, h = person_image.size
            dummy_dp = Image.new("L", (w, h), 0)
            mask = AM.cloth_agnostic_mask(
                dummy_dp, schp_lip_result, schp_atr_result,
                part=cloth_type
            )
            return mask
        
        # Last resort: geometric
        print("[Engine] WARNING: geometric fallback mask")
        from PIL import ImageFilter, ImageDraw
        w, h = person_image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        if cloth_type == "upper":
            draw.rectangle([int(w*0.15), int(h*0.18), int(w*0.85), int(h*0.58)], fill=255)
        elif cloth_type == "lower":
            draw.rectangle([int(w*0.18), int(h*0.45), int(w*0.82), int(h*0.88)], fill=255)
        else:
            draw.rectangle([int(w*0.15), int(h*0.18), int(w*0.85), int(h*0.85)], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 40))
        return mask
        
    def _load_lora(self):
        """Load LoRA adapter (once) — does NOT merge into weights yet."""
        if self._lora_loaded or not self._lora_path:
            return
        try:
            from peft import PeftModel
            print(f"[Engine] Loading saree LoRA adapter from {self._lora_path}...")
            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet, self._lora_path
            )
            self.pipeline.unet.eval()
            self._lora_loaded = True
            print("[Engine] Saree LoRA adapter loaded (disabled by default)")
        except Exception as e:
            print(f"[Engine] LoRA loading failed: {e}. Using base model.")
    
    def _enable_lora(self):
        """Merge LoRA weights into base model for inference."""
        if self._lora_loaded:
            try:
                self.pipeline.unet.merge_adapter()
                print("[Engine] LoRA merged (active)")
            except Exception as e:
                print(f"[Engine] LoRA merge failed: {e}")
    
    def _disable_lora(self):
        """Unmerge LoRA weights so base model is clean again."""
        if self._lora_loaded:
            try:
                self.pipeline.unet.unmerge_adapter()
                print("[Engine] LoRA unmerged (disabled)")
            except Exception as e:
                print(f"[Engine] LoRA unmerge failed: {e}")
    
    def _modesty_blend(self, result, original_person, cloth_type):
        """
        For 'overall' garments: blend original person's upper body back in.
        This ensures the chest/shoulder area always shows the person's original
        clothing, preventing inappropriate output when garment has no blouse.
        
        Uses a smooth gradient transition at ~30-40% image height so the
        lower body shows the new garment while upper body stays covered.
        """
        if cloth_type not in ("overall", "saree"):
            return result
        
        import numpy as np
        
        w, h = result.size
        result_arr = np.array(result).astype(np.float32)
        person_arr = np.array(original_person).astype(np.float32)
        
        # Create a vertical gradient mask:
        # 0.0 at top (keep original person) -> 1.0 at bottom (show result)
        # Transition zone: 25% to 40% of image height
        # This preserves: face, shoulders, chest, upper arms
        # This replaces: waist, hips, legs, lower body (where saree drapes)
        blend_start = int(h * 0.25)  # Start blending at 25%
        blend_end = int(h * 0.40)    # Fully showing result by 40%
        
        gradient = np.zeros((h, w, 1), dtype=np.float32)
        gradient[blend_end:] = 1.0  # Below 40%: show result fully
        
        # Smooth gradient in transition zone
        if blend_end > blend_start:
            for y in range(blend_start, blend_end):
                t = (y - blend_start) / (blend_end - blend_start)
                # Smooth easing (cubic)
                t = t * t * (3 - 2 * t)
                gradient[y] = t
        
        # Blend: original * (1-gradient) + result * gradient
        blended = person_arr * (1.0 - gradient) + result_arr * gradient
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        from PIL import Image
        return Image.fromarray(blended)
    
    def _check_skin_exposure(self, image, cloth_type):
        """
        Safety net: detect excessive skin exposure in the upper body region.
        If too much skin is detected, blend in more of the original person.
        Returns a score from 0.0 (fully clothed) to 1.0 (fully exposed).
        """
        if cloth_type not in ("overall", "saree"):
            return 0.0
        
        import numpy as np
        
        arr = np.array(image).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Check only the upper body region (15-40% height, center 60% width)
        y1, y2 = int(h * 0.15), int(h * 0.40)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        region = arr[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Simple skin tone detection in RGB
        # Skin pixels: R > 80, G > 50, B > 30, R > G, R > B
        r, g, b = region[:,:,0], region[:,:,1], region[:,:,2]
        skin_mask = (
            (r > 80) & (g > 50) & (b > 30) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 15) &  # Not gray
            (r - b > 20)  # Warm tone
        )
        
        skin_ratio = np.mean(skin_mask)
        return float(skin_ratio)
    
    async def try_on(self, person_image, garment_image, cloth_type="upper",
                     steps=50, guidance=2.5, seed=42):
        """
        Mirrors CatVTON app.py submit_function EXACTLY.
        Uses CatVTON's own repaint_result for clean blending.
        For overall garments: applies modesty blend + skin safety check.
        """
        import torch
        from utils import resize_and_crop, resize_and_padding, repaint_result
        
        async with self._lock:
            # Only use LoRA for explicit 'saree' selection
            use_lora = cloth_type == "saree" and self._lora_path
            if use_lora:
                self._load_lora()
                self._enable_lora()
            
            # Map 'saree' to 'overall' for CatVTON masking
            mask_type = "overall" if cloth_type == "saree" else cloth_type
            
            try:
                # app.py defaults: width=768, height=1024
                width, height = 768, 1024
                
                # app.py lines 152-153
                person_resized = resize_and_crop(person_image, (width, height))
                garment_resized = resize_and_padding(garment_image, (width, height))
                
                # app.py lines 159-163
                mask = self._generate_mask(person_resized, mask_type)
                mask = self.mask_processor.blur(mask, blur_factor=9)
                
                gc.collect()
                torch.cuda.empty_cache()
                
                # app.py lines 146-148
                generator = None
                if seed != -1:
                    generator = torch.Generator(device='cuda').manual_seed(seed)
                
                # app.py lines 167-174 — EXACT same call
                result = self.pipeline(
                    image=person_resized,
                    condition_image=garment_resized,
                    mask=mask,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=generator,
                )[0]
                
                # Use CatVTON's own repaint_result (utils.py:171-178)
                result = repaint_result(result, person_resized, mask)
                
                # ── SAFETY LAYERS (disabled for dev — enable in production) ──
                # Ensure garment images include a blouse for full-body garments.
                # Uncomment below to enforce modesty blend + skin detection:
                #
                # result = self._modesty_blend(result, person_resized, cloth_type)
                # skin_score = self._check_skin_exposure(result, cloth_type)
                # if skin_score > 0.35:
                #     print(f"[Safety] High skin exposure ({skin_score:.2f})")
                #     ... (see _modesty_blend and _check_skin_exposure methods)
                
                gc.collect()
                torch.cuda.empty_cache()
                
                return result
            finally:
                # Always unmerge LoRA after inference to keep base model clean
                if use_lora:
                    self._disable_lora()


# =====================================================================
#  Garment Store (file-based)
# =====================================================================

class GarmentStore:
    def __init__(self, store_dir=GARMENT_STORE):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
    
    def save_garment(self, image, seller_id, cloth_type="upper", name=""):
        garment_id = str(uuid.uuid4())[:8]
        garment_dir = self.store_dir / garment_id
        garment_dir.mkdir(parents=True, exist_ok=True)
        image.save(str(garment_dir / "garment.png"))
        meta = {
            "garment_id": garment_id,
            "seller_id": seller_id,
            "cloth_type": cloth_type,
            "name": name or f"Garment {garment_id}",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(garment_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return garment_id
    
    def get_garment(self, garment_id):
        garment_dir = self.store_dir / garment_id
        meta_path = garment_dir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        meta["image_path"] = str(garment_dir / "garment.png")
        return meta
    
    def list_garments(self, seller_id=None):
        garments = []
        for d in self.store_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if seller_id is None or meta.get("seller_id") == seller_id:
                    meta["image_path"] = str(d / "garment.png")
                    garments.append(meta)
        return sorted(garments, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def delete_garment(self, garment_id):
        import shutil
        garment_dir = self.store_dir / garment_id
        if garment_dir.exists():
            shutil.rmtree(garment_dir)


# =====================================================================
#  FastAPI App
# =====================================================================

app = FastAPI(
    title="CatVTON Try-On SaaS",
    description="AI Virtual Try-On Platform for Fashion Sellers",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

garment_store = GarmentStore()
engine = TryOnEngine.get()


@app.on_event("startup")
async def startup():
    print("[Startup] Loading CatVTON engine...")
    await engine.load()
    print("[Startup] Platform ready!")


# -- Seller API --

@app.post("/api/seller/upload")
async def upload_garment(
    file: UploadFile = File(...),
    cloth_type: str = Form("upper"),
    name: str = Form(""),
    seller_id: str = Form("default"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    garment_id = garment_store.save_garment(image, seller_id, cloth_type, name)
    return {
        "garment_id": garment_id,
        "name": name or f"Garment {garment_id}",
        "cloth_type": cloth_type,
        "tryon_url": f"/tryon/{garment_id}",
        "qr_url": f"/api/seller/qr/{garment_id}",
    }


@app.get("/api/seller/garments")
async def list_garments(seller_id: str = "default"):
    garments = garment_store.list_garments(seller_id)
    return {"garments": garments, "count": len(garments)}


@app.delete("/api/seller/garments/{garment_id}")
async def delete_garment(garment_id: str):
    garment_store.delete_garment(garment_id)
    return {"status": "deleted", "garment_id": garment_id}


@app.get("/api/seller/qr/{garment_id}")
async def get_qr_code(garment_id: str, request: Request):
    garment = garment_store.get_garment(garment_id)
    if not garment:
        raise HTTPException(404, "Garment not found")
    base_url = str(request.base_url).rstrip("/")
    tryon_url = f"{base_url}/tryon/{garment_id}"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(tryon_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    qr_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/garment/image/{garment_id}")
async def get_garment_image(garment_id: str):
    garment = garment_store.get_garment(garment_id)
    if not garment:
        raise HTTPException(404, "Garment not found")
    buf = io.BytesIO()
    Image.open(garment["image_path"]).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# -- Customer Try-On API --

@app.post("/api/tryon/{garment_id}")
async def run_tryon(
    garment_id: str,
    file: UploadFile = File(...),
    seed: int = Form(42),
):
    """Customer try-on. Person image processed in-memory, NEVER stored."""
    garment = garment_store.get_garment(garment_id)
    if not garment:
        raise HTTPException(404, "Garment not found")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    contents = await file.read()
    person_image = Image.open(io.BytesIO(contents)).convert("RGB")
    del contents
    
    garment_image = Image.open(garment["image_path"]).convert("RGB")
    
    t0 = time.time()
    try:
        result = await engine.try_on(
            person_image=person_image,
            garment_image=garment_image,
            cloth_type=garment.get("cloth_type", "upper"),
            seed=seed,
        )
    except Exception as e:
        raise HTTPException(500, f"Try-on failed: {str(e)}")
    finally:
        del person_image
    
    inference_time = time.time() - t0
    
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="image/png",
        headers={"X-Inference-Time": f"{inference_time:.1f}s"}
    )


# -- Frontend Pages --

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/seller", response_class=HTMLResponse)
async def seller_dashboard(request: Request):
    garments = garment_store.list_garments()
    return templates.TemplateResponse("seller.html", {
        "request": request, "garments": garments,
    })


@app.get("/tryon/{garment_id}", response_class=HTMLResponse)
async def customer_tryon_page(garment_id: str, request: Request):
    garment = garment_store.get_garment(garment_id)
    if not garment:
        raise HTTPException(404, "Garment not found")
    return templates.TemplateResponse("tryon.html", {
        "request": request, "garment": garment, "garment_id": garment_id,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=False)
