"""
Phase 1 Verification Script
Run this to confirm the environment is correctly set up.

Usage:
    conda activate catvton
    python scripts/verify_setup.py
"""
import sys
import os
import importlib

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check(name, import_name=None, version_attr="__version__"):
    """Check if a package is importable and print its version."""
    if import_name is None:
        import_name = name
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, version_attr, "unknown")
        print(f"  [OK] {name}: {ver}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name}: NOT FOUND ({e})")
        return False


def main():
    print("=" * 50)
    print("  Phase 1 -- Environment Verification")
    print("=" * 50)
    print()

    # Python version
    print(f"[Python] {sys.version}")
    py_ok = sys.version_info[:2] == (3, 9)
    if not py_ok:
        print(f"  [WARN] Expected Python 3.9, got {sys.version_info[0]}.{sys.version_info[1]}")
    print()

    # Core packages
    print("[Core ML Packages]")
    all_ok = True
    all_ok &= check("torch")
    all_ok &= check("torchvision")
    all_ok &= check("diffusers")
    all_ok &= check("transformers")
    all_ok &= check("accelerate")
    all_ok &= check("xformers")
    print()

    # CUDA check
    print("[CUDA]")
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024 ** 3)
            print(f"  [OK] CUDA available: {gpu} ({mem_gb:.1f} GB)")
            print(f"  [OK] CUDA version: {torch.version.cuda}")

            # Quick VRAM allocation test (fp16)
            t = torch.randn(1, 4, 64, 48, device="cuda", dtype=torch.float16)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(f"  [OK] FP16 tensor test: {allocated:.1f} MB allocated")
            del t
            torch.cuda.empty_cache()
        else:
            print("  [FAIL] CUDA NOT available")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] CUDA check failed: {e}")
        all_ok = False
    print()

    # Utility packages
    print("[Utility Packages]")
    all_ok &= check("cv2", "cv2", "__version__")
    all_ok &= check("PIL", "PIL", "__version__")
    all_ok &= check("numpy")
    all_ok &= check("scipy")
    all_ok &= check("yaml", "yaml")
    all_ok &= check("gradio")
    print()

    # CatVTON repo check
    print("[CatVTON Repo]")
    from pathlib import Path
    catvton_dir = Path(__file__).parent.parent / "CatVTON"
    if (catvton_dir / "app.py").exists():
        print(f"  [OK] CatVTON repo found at {catvton_dir}")
    else:
        print(f"  [FAIL] CatVTON repo NOT found at {catvton_dir}")
        all_ok = False

    if (catvton_dir / "inference.py").exists():
        print(f"  [OK] inference.py present")
    else:
        print(f"  [FAIL] inference.py NOT found")
        all_ok = False
    print()

    # Project structure
    print("[Project Structure]")
    project_root = Path(__file__).parent.parent
    required = [
        "app/__init__.py",
        "app/config.py",
        "app/preprocessing/__init__.py",
        "app/pipeline/__init__.py",
        "app/api/__init__.py",
        "configs/rtx4050.yaml",
        "requirements.txt",
    ]
    for f in required:
        if (project_root / f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [FAIL] {f} MISSING")
            all_ok = False
    print()

    # Summary
    print("=" * 50)
    if all_ok:
        print("  ALL CHECKS PASSED -- Environment ready!")
    else:
        print("  SOME CHECKS FAILED -- See above for details")
    print("=" * 50)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
