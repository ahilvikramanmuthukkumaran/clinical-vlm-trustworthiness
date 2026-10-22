"""Step 0: check that your laptop can run the model stack.

Run:  python model/scripts/check_env.py
Paste the full output back to Claude.
"""
import importlib
import platform
import sys


def version_of(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed (no version)")
    except Exception as exc:  # noqa: BLE001
        return f"NOT AVAILABLE ({type(exc).__name__}: {exc})"


print("Python        :", sys.version.split()[0])
print("Platform      :", platform.platform())
for pkg in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "PIL"]:
    print(f"{pkg:<14}:", version_of(pkg))

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU           :", props.name)
        print("VRAM (GB)     :", round(props.total_memory / 1024**3, 2))
        print("bf16 support  :", torch.cuda.is_bf16_supported())
except Exception as exc:  # noqa: BLE001
    print("torch check failed:", exc)
