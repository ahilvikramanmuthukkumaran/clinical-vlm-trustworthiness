# Hardware Fit Test (Stage 1)

Date: 2026-10-22. Owner: model development.

Question: which base VLM can this laptop actually run, and can it also *train* one?

**Answer: use Qwen2-VL-2B-Instruct on the laptop.** MedGemma-4B runs for inference
but cannot be LoRA-trained here; if MedGemma is wanted, training has to move to
Kaggle. Details and numbers below.

---

## 1. Environment

All of this was verified by running, not assumed.

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language 10.0.26300 |
| CPU / RAM | Intel i7 / 16 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | **4096 MiB (4.00 GB)**, compute capability 8.6 |
| Driver | 617.14, CUDA UMD 13.4 |
| Python | 3.13.13 in `.venv` (repo root, git-ignored) |
| torch | 2.14.1+cu126, `torch.cuda.is_available()` = **True** |
| bf16 | supported |
| transformers | 5.20.2 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |
| pillow | 12.3.0 |

PyTorch was installed from the cu126 index. That index carries newer builds than
cu128 (2.14.1 vs 2.11.0) and the installed driver is far ahead of both, so cu126
is the better choice here rather than a compatibility compromise.

**bitsandbytes works on Windows.** Not just the import: a real
`bnb.nn.Linear4bit(256, 256, quant_type="nf4")` forward pass ran on the GPU and
returned finite bf16 output. `bnb.optim.PagedAdamW8bit` is present (though the
tests below used plain `torch.optim.AdamW`).

### Two environment facts that matter

**1. The desktop already holds ~0.77 GB of VRAM.** Measured with
`torch.cuda.mem_get_info()`: only **3.23 GB of the 4.00 GB was free** before
loading anything. Every budget below is against ~3.2 GB usable, not 4.0 GB.
Closing browsers and other GPU apps is the cheapest way to buy headroom.

**2. `datasets` is broken on this machine, and it is not a Python problem.**
Smart App Control is enabled (`VerifiedAndReputablePolicyState = 1`) and blocks
`pyarrow`'s unsigned `_compute` DLL:

```
ImportError: DLL load failed while importing _compute:
An Application Control policy has blocked this file.
```

This takes down `pyarrow.compute`, `pandas` 3.0.6 (which now hard-depends on
pyarrow) and `datasets` 5.1.0. It did **not** affect the fit test — torch,
transformers, peft and bitsandbytes are all fine — but it will block Stage 2
data work. Two ways out, neither applied yet because both are the owner's call:

- Pin `pandas<3` and skip `datasets` entirely. The test split is plain JSON
  (`data/splits/test.json`, see `INTERFACE.md`), so Arrow is not actually needed.
  This is the lower-risk option and touches no system settings.
- Turn Smart App Control off. Note this is **irreversible without reinstalling
  Windows** — it cannot be switched back on. Not recommended just for this.

---

## 2. Test runs

Every row below was executed on this laptop. "Peak VRAM (torch)" is
`torch.cuda.max_memory_allocated()`; "Peak (GPU)" is the highest
`nvidia-smi memory.used` sampled during the run.

### Inference — `model/scripts/test_load.py`

| Model | Result | Load | Peak VRAM (torch) | Peak (GPU) | Generate | Max temp |
|---|---|---|---|---|---|---|
| MedGemma-4B, `device_map="auto"` | **failed** | — | — | 8 MiB | — | 40 °C |
| MedGemma-4B, forced to GPU 0 | **works** | 20 s | 3.14 GB | 3359 MiB | 2.4 s | 48 °C |
| Qwen2-VL-2B, `device_map="auto"` | **works** | 8 s | 1.50 GB | 1583 MiB | 1.3 s | 43 °C |

Both working models answered the prompt ("Is there a fracture visible in this
image?" on a blank grey image) with `No`. The answer is meaningless — the image
is blank — it only proves the forward pass completes.

The MedGemma `auto` failure was **not** a real out-of-memory. accelerate reserves
headroom for activations, decided the model would not fit, and raised:

```
ValueError: Some modules are dispatched on the CPU or the disk.
```

Forcing `device_map={"": 0}` removes that reserve and the model loads fine.
`test_load.py` now catches exactly this error and retries forced-to-GPU, so the
script works unmodified on this hardware. It also prints device placement and
warns if anything lands on CPU/disk, so a spilled load can't be mistaken for a fit.

**Why MedGemma is so tight.** Of its 3.01 GB 4-bit footprint, only 1.74 GB is
quantised weights. **1.26 GB is the embedding table**, which bitsandbytes leaves
in bf16 (Gemma 3's 262k vocab × 2560). After loading, **0.18 GB of VRAM was
free.** Qwen2-VL-2B, with a 152k vocab and 1536 width, needs 1.42 GB total.

### Training — `model/scripts/test_train_memory.py`

4-bit base + LoRA r on attention projections (`q_proj,k_proj,v_proj,o_proj`)
only, gradient checkpointing on, batch size 1, dummy image + short text.

| Model | Image | LoRA r | Tokens | Peak VRAM (torch) | Peak (GPU) | s/step | Fits? |
|---|---|---|---|---|---|---|---|
| MedGemma-4B | 448 | 8 | 281 | 6.09 GB | 3953 MiB | 148 | **no** |
| MedGemma-4B | 224 | 4 | 281 | 6.03 GB | 3951 MiB | 82 | **no** |
| Qwen2-VL-2B | 448 | 8 | 290 | 2.78 GB | 3055 MiB | 1.7 | **yes** |
| Qwen2-VL-2B | 448 | 16 | 290 | 2.80 GB | 3097 MiB | 1.7 | **yes** |
| Qwen2-VL-2B | 560 | 8 | 434 | 3.23 GB | 3553 MiB | 3.1 | **yes** |
| Qwen2-VL-2B | 672 | 8 | 610 | 3.77 GB | 3955 MiB | 6.2 | yes, no margin |
| Qwen2-VL-2B | 896 | 8 | 1058 | 5.17 GB | 3953 MiB | 15.1 | **no** |
| Qwen2-VL-2B | 896 | 16 | 1058 | 5.19 GB | 3953 MiB | 15.3 | **no** |

Loss decreased across steps in every run that fit (Qwen 448/r8: 4.24 → 4.08 →
3.94), so the backward pass and optimiser step are really happening.

### How "does not fit" was detected — this matters

**None of these runs ever raised CUDA out-of-memory.** On Windows WDDM the NVIDIA
driver silently falls back to shared system RAM, so an over-budget run *completes*
and looks like a pass. The giveaways are that peak torch allocation exceeds
physical VRAM (6.09 GB on a 4.00 GB card) while `nvidia-smi` pins at ~3.95 GB, and
that step time collapses — 148 s/step for MedGemma versus 1.7 s for Qwen.

I tried to force an honest OOM with
`torch.cuda.set_per_process_memory_fraction(1.0)` and **measured that it does not
work**: the fallback happens in the driver, `cudaMalloc` still succeeds from shared
memory, and torch never sees a failure. The `--hard-cap` flag is kept but its help
text now records this. Only the NVIDIA Control Panel setting
"CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback" can actually disable
the spill; that is a system setting and was left alone.

So `test_train_memory.py` now judges fit by comparing peak allocation against
physical VRAM and by step time, and says "runs, slowly" instead of "fits" when it
sees a spill.

**MedGemma training cannot be shrunk into this card.** Dropping the image to 224 px
and LoRA to r=4 changed peak VRAM by 0.06 GB (6.09 → 6.03 GB) and left the sequence
at **281 tokens, identical to 448 px**. Gemma 3's vision tower is fixed at 896×896
internally, so `--image-size` is inert, and the loss over a 262k-token vocab is a
fixed cost. There is no setting that makes it fit; 41 hours per 1000 steps is the
best case observed.

### Largest Qwen settings that worked

- **448 px, r=8 or r=16 — recommended.** 2.78–2.80 GB, 1.7 s/step (~0.5 h / 1000
  steps). Comfortable margin. LoRA rank is nearly free (+0.02 GB for r=16), so
  prefer r=16 over a bigger image if capacity is needed.
- **560 px, r=8 — safe ceiling.** 3.23 GB, 3.1 s/step (~0.9 h / 1000 steps).
- **672 px, r=8 — the largest that technically fit,** but `nvidia-smi` hit
  3955 of 4096 MiB (96.6%). Do not rely on this: real images with varied aspect
  ratios will produce more tokens than a square dummy and tip it over.
- **896 px — does not fit** at any rank tested.

---

## 3. Temperature

The GPU was sampled every 3 s throughout every run by a wrapper that kills the
process immediately if it passes 88 °C.

- **552 samples across 14 runs. Hottest reading: 69 °C.**
- Highest per-run means: MedGemma 448/r8 61 °C, Qwen 896/r16 59 °C.
- The 88 °C kill-switch **never fired**. No thermal warning is warranted.

Caveat: the longest run was about 7 minutes. Sustained multi-hour training will
get hotter than anything measured here.

---

## 4. Recommendation

**Qwen2-VL-2B-Instruct on the laptop**, 4-bit + LoRA r=8–16 on attention
projections, batch size 1, gradient checkpointing, images at 448–560 px.

Reasoning:

- It is the only option where **both** inference and training fit in real VRAM
  with margin, at a workable 1.7–3.1 s/step.
- MedGemma-4B is usable on this laptop **for inference only** (3.14 GB peak), and
  even then with just 0.18 GB free — likely too tight to run alongside the
  Streamlit app on the same GPU, which is worth checking before relying on it.
- MedGemma-4B training on this laptop is not viable: 41 h/1000 steps, spilling to
  system RAM, and not fixable by lowering settings.

If the project specifically needs MedGemma's medical pretraining, the split is:
**train the LoRA adapter on Kaggle** (16 GB T4/P100 — ample for a 3.01 GB base),
then run inference locally. That costs a dependency on Kaggle availability and
makes the laptop unable to reproduce training. Nothing about Kaggle was tested
here. An honest comparison would be to fine-tune Qwen2-VL-2B first, since it is
fully reproducible locally, and treat MedGemma as the stretch goal.

---

## 5. What is verified, and what is not

### Verified by running on this laptop

- CUDA, bf16 and a real 4-bit bitsandbytes matmul all work.
- MedGemma-4B loads in 4-bit and generates (forced to GPU 0); 3.14 GB peak.
- Qwen2-VL-2B loads in 4-bit and generates; 1.50 GB peak.
- Qwen2-VL-2B LoRA training runs inside real VRAM at 448–672 px, with decreasing loss.
- MedGemma-4B LoRA training exceeds VRAM and spills, at every setting tried.
- Peak VRAM and step times in the tables above.
- GPU stayed at or below 69 °C.
- `datasets`/`pandas`/`pyarrow.compute` are blocked by Smart App Control.
- `set_per_process_memory_fraction` does not prevent the WDDM sysmem spill.

### Not verified — open questions

- **Nothing about model quality.** No clinical accuracy, no hallucination
  measurement, no comparison between the two models. The only generation tested
  was one blank grey image, and "No" proves only that the forward pass completes.
  Qwen2-VL-2B is a general model with no medical pretraining; whether it is good
  enough for this project is completely untested and is the main risk in this
  recommendation.
- **Real images.** All tests used a flat grey square. Qwen2-VL tokenises images
  dynamically, so real X-rays at non-square aspect ratios will produce different
  token counts — and therefore different memory — than the table shows. The
  672 px row in particular has no margin to absorb that.
- **Real training targets.** The dummy answer was "No." (a few tokens). Longer
  supervision will lengthen sequences and raise memory above these numbers.
- **Batch size > 1 and gradient accumulation** were never tested; only batch size 1.
- **Sustained training.** Longest run was ~7 minutes and 3 steps. Throughput,
  thermals and stability over thousands of steps are unknown, as is whether
  step time stays at 1.7 s once a real data loader is involved.
- **Which modules LoRA actually attached to.** `target_modules` matches by name
  suffix, so it may have wrapped vision-tower attention as well as the language
  model (MedGemma reported 5.95 M trainable params, Qwen 2.18 M). This should be
  pinned down deliberately before real training.
- **Kaggle** was not touched. The MedGemma-on-Kaggle path is reasoning, not measurement.
- **8-bit / paged optimisers** were not used; plain AdamW. `PagedAdamW8bit` is
  available and would save a little more.
- **The data pipeline**, blocked by the `datasets` issue above.

---

## 6. Reproducing

```bash
python model/scripts/check_env.py
python model/scripts/test_load.py                              # MedGemma-4B
python model/scripts/test_load.py Qwen/Qwen2-VL-2B-Instruct
python model/scripts/test_train_memory.py Qwen/Qwen2-VL-2B-Instruct --image-size 448 --lora-r 8
```

MedGemma-4B is gated: accept its licence on Hugging Face and run
`huggingface-cli login` once first. Its weights are 8.01 GB on disk.
