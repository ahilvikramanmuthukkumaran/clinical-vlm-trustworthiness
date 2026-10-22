"""Smoke test for the real predict() implementation.

Runs both modes on one image and one question and prints the results, peak VRAM
and timing. Does not train anything.

Usage:
    python model/scripts/smoke_predict.py
    python model/scripts/smoke_predict.py path/to/image.png "Is there a fracture?"
    python model/scripts/smoke_predict.py --adapter model/checkpoints/my_adapter

With no image argument it uses data/samples/synthetic_cxr.png, generating a
crude synthetic chest-film-like image if that file does not exist yet. A
synthetic image is only good enough to prove the pipeline runs - it says nothing
about clinical accuracy.
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from model.src.predict import cuda_available, load_config, predict  # noqa: E402

DEFAULT_IMAGE = REPO_ROOT / "data" / "samples" / "synthetic_cxr.png"
DEFAULT_QUESTION = "Is there a rib fracture visible in this image?"


def make_synthetic_cxr(path: Path, size: int = 448) -> Path:
    """Write a crude greyscale chest-film-like image: dark field, two lung ovals, ribs.

    This is a placeholder so the smoke test has a non-blank image with some
    structure. It is not a medical image and carries no ground truth.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (size, size), 18)
    draw = ImageDraw.Draw(img)

    # Thorax silhouette
    draw.ellipse([size * 0.12, size * 0.10, size * 0.88, size * 0.92], fill=70)
    # Two darker lung fields
    draw.ellipse([size * 0.18, size * 0.20, size * 0.46, size * 0.78], fill=38)
    draw.ellipse([size * 0.54, size * 0.20, size * 0.82, size * 0.78], fill=38)
    # Spine
    draw.rectangle([size * 0.47, size * 0.14, size * 0.53, size * 0.88], fill=105)
    # Ribs
    for i in range(7):
        y = size * (0.24 + i * 0.075)
        draw.arc([size * 0.16, y - size * 0.07, size * 0.50, y + size * 0.07],
                 start=200, end=340, fill=95, width=3)
        draw.arc([size * 0.50, y - size * 0.07, size * 0.84, y + size * 0.07],
                 start=200, end=340, fill=95, width=3)

    img = img.filter(ImageFilter.GaussianBlur(1.2)).convert("RGB")
    img.save(path)
    return path


def vram_report(torch_mod) -> str:
    if torch_mod is None:
        return "n/a (no CUDA)"
    return (f"alloc {torch_mod.cuda.memory_allocated()/1024**3:.2f} GB, "
            f"peak {torch_mod.cuda.max_memory_allocated()/1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--adapter", default=None,
                        help="optional LoRA adapter directory passed to predict()")
    args = parser.parse_args()

    torch_mod = None
    if cuda_available():
        import torch
        torch_mod = torch
        torch_mod.cuda.reset_peak_memory_stats()

    config = load_config()
    print("=" * 68)
    print("smoke_predict")
    print("=" * 68)
    print(f"CUDA available       : {cuda_available()}")
    print(f"model_id             : {config['model_id']}")
    print(f"abstention_threshold : {config['abstention_threshold']}")
    print(f"max_new_tokens       : {config['max_new_tokens']}")
    print(f"adapter_path         : {args.adapter or 'None'}")

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise SystemExit(f"image not found: {image_path}")
        origin = "supplied"
    else:
        image_path = DEFAULT_IMAGE
        if image_path.exists():
            origin = "existing synthetic"
        else:
            make_synthetic_cxr(image_path)
            origin = "newly generated synthetic"

    with Image.open(image_path) as probe:
        print(f"image                : {image_path.relative_to(REPO_ROOT)} "
              f"({origin}, {probe.size[0]}x{probe.size[1]})")
    print(f"question             : {args.question!r}")

    results = {}
    for mode in ("baseline", "mitigated"):
        print()
        print("-" * 68)
        print(f"mode = {mode}")
        print("-" * 68)
        t0 = time.time()
        result = predict(image_path, args.question, mode=mode, adapter_path=args.adapter)
        elapsed = time.time() - t0
        results[mode] = result

        print(f"answer      : {result.answer!r}")
        print(f"confidence  : {result.confidence:.4f}")
        print(f"abstained   : {result.abstained}")
        print(f"is_stub     : {result.is_stub}")
        print(f"raw_text    : {result.raw_text!r}")
        print(f"time        : {elapsed:.1f}s"
              + ("  (includes model load)" if mode == "baseline" else ""))
        print(f"VRAM        : {vram_report(torch_mod)}")

    print()
    print("=" * 68)
    print("to_dict() keys:", sorted(results['baseline'].to_dict().keys()))
    if torch_mod is not None:
        print(f"Peak VRAM overall   : "
              f"{torch_mod.cuda.max_memory_allocated()/1024**3:.2f} GB alloc / "
              f"{torch_mod.cuda.max_memory_reserved()/1024**3:.2f} GB reserved")
        print(f"GPU                 : {torch_mod.cuda.get_device_name(0)} "
              f"({torch_mod.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB)")
    else:
        print("Ran on the STUB path - no GPU, so nothing about the real model was tested.")
    print("=" * 68)


if __name__ == "__main__":
    main()
