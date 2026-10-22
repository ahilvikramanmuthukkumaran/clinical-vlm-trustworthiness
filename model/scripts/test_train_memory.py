"""Step 2: can this laptop TRAIN the model, not just run it?

Loads the model in 4-bit, attaches a small LoRA adapter (attention projections
only), turns on gradient checkpointing and runs a few forward+backward steps on
a dummy image + short prompt with batch size 1. Prints peak VRAM and the hottest
GPU temperature seen while training.

Usage:
    python model/scripts/test_train_memory.py
    python model/scripts/test_train_memory.py Qwen/Qwen2-VL-2B-Instruct
    python model/scripts/test_train_memory.py Qwen/Qwen2-VL-2B-Instruct --image-size 224 --lora-r 4

Every knob that costs memory is a flag, so a run that dies with out-of-memory
can be retried smaller and the largest setting that fits can be reported.
"""
import argparse
import subprocess
import threading
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Attention projections only: the smallest sensible LoRA surface.
ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_id", nargs="?", default="google/medgemma-4b-it")
    p.add_argument("--image-size", type=int, default=448,
                   help="side length of the dummy square image fed to the processor")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="cap on image pixels for processors that tokenise images dynamically "
                        "(Qwen2-VL); ignored by fixed-resolution vision towers (Gemma 3)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--max-len", type=int, default=None,
                   help="truncate the tokenised sequence to this many tokens")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--hard-cap", action="store_true",
                   help="ask torch to cap the allocator at real VRAM. MEASURED NOT TO WORK on "
                        "this Windows/WDDM setup: the sysmem fallback happens in the driver, so "
                        "cudaMalloc still succeeds from shared system RAM and torch never sees "
                        "an OOM. Judge fit by comparing peak allocation against physical VRAM "
                        "and by step time instead; only the NVIDIA Control Panel setting "
                        "'CUDA - Sysmem Fallback Policy' can actually turn the spill off")
    return p.parse_args()


class GpuProbe(threading.Thread):
    """Samples nvidia-smi in the background so we can report heat and VRAM."""

    daemon = True

    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.temps = []
        self.used_mb = []
        self._stop = threading.Event()

    def run(self):
        query = "--query-gpu=temperature.gpu,memory.used"
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                temp, used = out.stdout.strip().splitlines()[0].split(",")
                self.temps.append(int(temp))
                self.used_mb.append(int(used))
            except Exception:  # noqa: BLE001 - a missed sample must not kill the test
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)


def build_processor(args):
    kwargs = {}
    if args.max_pixels is not None:
        # Qwen2-VL style dynamic resolution; harmless to pass, so guarded by a retry.
        kwargs["max_pixels"] = args.max_pixels
    try:
        return AutoProcessor.from_pretrained(args.model_id, **kwargs)
    except TypeError:
        print("processor does not accept max_pixels; loading without it")
        return AutoProcessor.from_pretrained(args.model_id)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - this test is meaningless on CPU")

    print(f"Model        : {args.model_id}")
    print(f"Image size   : {args.image_size}px square"
          + (f" (max_pixels={args.max_pixels})" if args.max_pixels else ""))
    print(f"LoRA rank    : {args.lora_r} on {ATTENTION_PROJECTIONS}")
    print(f"Max seq len  : {args.max_len or 'untruncated'}")
    print(f"Steps        : {args.steps} (batch size 1)")
    print(f"Hard cap     : {'on - OOM instead of sysmem spill' if args.hard_cap else 'off'}")

    if args.hard_cap:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)

    probe = GpuProbe()
    probe.start()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    t0 = time.time()
    processor = build_processor(args)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    print(f"Loaded in {time.time() - t0:.0f}s")
    print(f"VRAM after load           : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=ATTENTION_PROJECTIONS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    image = Image.new("RGB", (args.image_size, args.image_size), (128, 128, 128))
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Is there a fracture visible in this image?"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "No."}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
    )
    if args.max_len is not None:
        inputs = {
            k: (v[:, : args.max_len]
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] > args.max_len else v)
            for k, v in inputs.items()
        }
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    # Image placeholders must not contribute to the loss.
    for attr in ("image_token_id", "image_token_index"):
        tok = getattr(model.config, attr, None)
        if isinstance(tok, int):
            labels[labels == tok] = -100
    inputs["labels"] = labels
    print(f"Sequence length fed       : {inputs['input_ids'].shape[1]} tokens")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=1e-4)

    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for step in range(1, args.steps + 1):
        t_step = time.time()
        out = model(**inputs)
        out.loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        step_times.append(time.time() - t_step)
        print(f"step {step}: loss={out.loss.item():.4f}  "
              f"{step_times[-1]:.1f}s  "
              f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    probe.stop()
    print(f"\nPeak VRAM (torch alloc)   : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak VRAM (torch reserved): {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    if probe.used_mb:
        print(f"Peak VRAM (whole GPU)     : {max(probe.used_mb) / 1024:.2f} GB of "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    if probe.temps:
        print(f"GPU temperature           : max {max(probe.temps)} C, "
              f"mean {sum(probe.temps) / len(probe.temps):.0f} C, {len(probe.temps)} samples")
        if max(probe.temps) > 88:
            print("WARNING: GPU went above 88 C - improve cooling before long training runs")

    median_step = sorted(step_times)[len(step_times) // 2]
    total_vram = torch.cuda.get_device_properties(0).total_memory
    spilled = torch.cuda.max_memory_allocated() > total_vram
    print(f"\nMedian step time         : {median_step:.1f}s "
          f"(~{median_step * 1000 / 3600:.1f} h per 1000 steps)")

    if spilled:
        print("RESULT: the steps completed but peak allocation EXCEEDED physical VRAM, so the "
              "driver spilled into shared system RAM. This is not a fit - it runs, slowly. "
              "Re-run with --hard-cap to see the honest out-of-memory point.")
    elif median_step > 20:
        print("RESULT: fits in VRAM, but each step is slow enough that real training "
              "would take days. Treat as 'runs', not 'practical'.")
    else:
        print("RESULT: fits in VRAM at a workable speed. Settings above are the ones that fit.")


if __name__ == "__main__":
    main()
