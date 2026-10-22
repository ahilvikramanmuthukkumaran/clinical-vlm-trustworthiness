"""Step 1: 10-minute fit test. Does the model load in 4-bit and answer one question?

Usage:
    python model/scripts/test_load.py                         # MedGemma-4B (default)
    python model/scripts/test_load.py Qwen/Qwen2-VL-2B-Instruct   # fallback model
    python model/scripts/test_load.py google/medgemma-4b-it path/to/xray.png

MedGemma is a gated model: accept the license on its Hugging Face page, then run
`huggingface-cli login` once before this script.
Close Chrome and other heavy apps first. Keep the laptop plugged in.
"""
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

model_id = sys.argv[1] if len(sys.argv) > 1 else "google/medgemma-4b-it"
image_path = sys.argv[2] if len(sys.argv) > 2 else None

print(f"Model: {model_id}")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

t0 = time.time()
processor = AutoProcessor.from_pretrained(model_id)
try:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
except ValueError as exc:
    # On a 4 GB card accelerate's "auto" map reserves headroom for activations and
    # then refuses to place the model at all. Forcing every module onto the GPU
    # removes that reserve; MedGemma-4B does fit this way, with little to spare.
    if "dispatched on the CPU or the disk" not in str(exc):
        raise
    print('device_map="auto" refused to place the model; retrying with everything on GPU 0')
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
model.eval()
print(f"Loaded in {time.time() - t0:.0f}s")
print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# device_map="auto" quietly spills to CPU/disk when VRAM runs short, which would
# make a model that does NOT fit look like a success. Say where the weights landed.
device_map = getattr(model, "hf_device_map", None)
placements = set(device_map.values()) if device_map else {p.device for p in model.parameters()}
print("Device placement:", placements or "{unknown}")
offloaded = {d for d in placements if str(d) in ("cpu", "disk")}
if offloaded:
    print(f"WARNING: part of the model was offloaded to {offloaded} - it does NOT fit in VRAM alone")

image = Image.open(image_path).convert("RGB") if image_path else Image.new("RGB", (448, 448), (128, 128, 128))
question = "Is there a fracture visible in this image? Answer yes or no."
messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
answer = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("Answer:", answer.strip())
print(f"Generation time: {time.time() - t0:.1f}s")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("RESULT: model loads and generates. Training memory is a separate test.")
