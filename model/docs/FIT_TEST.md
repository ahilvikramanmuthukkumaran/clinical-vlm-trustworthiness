# Hardware Fit Test (Stage 1)

Date: 2026-10-22. Owner: model development.

Question: which base VLM can this laptop actually run, and can it also *train* one?

**Decision: the project uses MedGemma-4B only, with no fallback to a general
model.** The laptop is used for MedGemma-4B **inference only**. Training and data
preparation run on a **cloud GPU (Kaggle/Colab)**, because local MedGemma training
does not fit this card. Details and numbers below; the decision is in §4.

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

MedGemma-4B answered the prompt ("Is there a fracture visible in this
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
free.** That 0.18 GB of headroom is the binding constraint on all local use.

### Training — `model/scripts/test_train_memory.py`

4-bit base + LoRA r on attention projections (`q_proj,k_proj,v_proj,o_proj`)
only, gradient checkpointing on, batch size 1, dummy image + short text.

| Model | Image | LoRA r | Tokens | Peak VRAM (torch) | Peak (GPU) | s/step | Fits? |
|---|---|---|---|---|---|---|---|
| MedGemma-4B | 448 | 8 | 281 | 6.09 GB | 3953 MiB | 148 | **no** |
| MedGemma-4B | 224 | 4 | 281 | 6.03 GB | 3951 MiB | 82 | **no** |

Neither MedGemma configuration fits. The reference measurements in §2.1 confirm
the test harness itself can register a genuine fit, so these two "no" results are
a property of the model on this card, not of the measurement.

### How "does not fit" was detected — this matters

**None of these runs ever raised CUDA out-of-memory.** On Windows WDDM the NVIDIA
driver silently falls back to shared system RAM, so an over-budget run *completes*
and looks like a pass. The giveaways are that peak torch allocation exceeds
physical VRAM (6.09 GB on a 4.00 GB card) while `nvidia-smi` pins at ~3.95 GB, and
that step time collapses — 148 s/step for MedGemma versus 1.7 s for a model that
genuinely fits (§2.1).

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

### 2.1 Reference measurement, not used in the project — Qwen2-VL-2B

> **These results are kept for reference only.** Qwen2-VL-2B-Instruct is **not
> used in this project** and is **not a fallback**. The project uses MedGemma-4B
> only (§4). These numbers are retained for two reasons: they prove the test
> harness can distinguish a genuine fit from a system-RAM spill, and they record
> what a 2B-class model costs on this hardware should the question ever be
> reopened. Nothing downstream should depend on them.

Qwen2-VL-2B-Instruct was measured as a smaller comparison model while the fit
question was still open. Same 4-bit + LoRA setup as above.

Inference — `model/scripts/test_load.py`:

| Model | Result | Load | Peak VRAM (torch) | Peak (GPU) | Generate | Max temp |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B, `device_map="auto"` | **works** | 8 s | 1.50 GB | 1583 MiB | 1.3 s | 43 °C |

It also answered `No` on the blank grey image — equally meaningless, equally only
a proof that the forward pass completes. Unlike MedGemma it needed no
`device_map` override: with a 152k vocab and 1536 width it needs 1.42 GB total.

Training — `model/scripts/test_train_memory.py`:

| Model | Image | LoRA r | Tokens | Peak VRAM (torch) | Peak (GPU) | s/step | Fits? |
|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B | 448 | 8 | 290 | 2.78 GB | 3055 MiB | 1.7 | **yes** |
| Qwen2-VL-2B | 448 | 16 | 290 | 2.80 GB | 3097 MiB | 1.7 | **yes** |
| Qwen2-VL-2B | 560 | 8 | 434 | 3.23 GB | 3553 MiB | 3.1 | **yes** |
| Qwen2-VL-2B | 672 | 8 | 610 | 3.77 GB | 3955 MiB | 6.2 | yes, no margin |
| Qwen2-VL-2B | 896 | 8 | 1058 | 5.17 GB | 3953 MiB | 15.1 | **no** |
| Qwen2-VL-2B | 896 | 16 | 1058 | 5.19 GB | 3953 MiB | 15.3 | **no** |

Loss decreased across steps in every run that fit (Qwen 448/r8: 4.24 → 4.08 →
3.94), so the backward pass and optimiser step were really happening.

Largest Qwen settings that worked:

- **448 px, r=8 or r=16.** 2.78–2.80 GB, 1.7 s/step (~0.5 h / 1000 steps).
  Comfortable margin. LoRA rank is nearly free (+0.02 GB for r=16).
- **560 px, r=8 — safe ceiling.** 3.23 GB, 3.1 s/step (~0.9 h / 1000 steps).
- **672 px, r=8 — the largest that technically fit,** but `nvidia-smi` hit
  3955 of 4096 MiB (96.6%). Real images with varied aspect
  ratios would produce more tokens than a square dummy and tip it over.
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

## 4. Decision

**The project uses MedGemma-4B (`google/medgemma-4b-it`) only. There is no
fallback to a general model.** Work is split across two machines:

### Cloud GPU (Kaggle/Colab) — training and data preparation

All LoRA training and all dataset preparation run on a cloud GPU. This is forced
by measurement, not preference: local MedGemma-4B LoRA training peaked at
**6.09 GB on a 4.00 GB card**, spilled into shared system RAM, and ran at about
**148 s/step** (~41 h per 1000 steps). It cannot be shrunk into the card either —
dropping to 224 px and r=4 moved peak VRAM by 0.06 GB and left the sequence at an
identical 281 tokens, because Gemma 3's vision tower is fixed at 896×896 and the
loss over a 262k-token vocab is a fixed cost.

Data preparation also moves to the cloud, which conveniently routes around the
Smart App Control problem in §1: `datasets`, `pandas` 3.x and `pyarrow.compute`
are all blocked locally, and a cloud notebook has a working Arrow stack.

A 16 GB T4 or P100 is ample for a 3.01 GB 4-bit base. **None of this was tested —
see §5.**

### Laptop (RTX 3050, 4 GB) — MedGemma-4B inference only

Verified working: **3.14 GB peak**, 20 s load, 2.4 s generation.

Two hard requirements for any local inference code:

- **`device_map={"": 0}` is mandatory.** `device_map="auto"` fails outright on
  this card — accelerate reserves activation headroom, decides the model will not
  fit, and raises `ValueError: Some modules are dispatched on the CPU or the disk`.
  `test_load.py` already handles this by catching that error and retrying forced
  to GPU 0; `model/src/predict.py` must do the same when the stub is replaced.
- **Only about 0.18 GB of VRAM is free after loading.** This is the tightest
  constraint in the project. Practical consequences: batch size stays at 1,
  generation lengths stay short, and the model very likely **cannot share the GPU
  with the Streamlit app** — that needs checking before the app is wired to a real
  local model, since 0.77 GB is already held by the desktop before anything loads.
  Closing other GPU applications is the cheapest available headroom.

### What this decision costs

Worth stating plainly, since it was chosen with the trade-off visible:

- Training is **not reproducible on the laptop**. The project now depends on
  Kaggle/Colab availability, session limits and upload/download of adapters.
- There is **no fallback model**. If MedGemma-4B turns out to answer poorly on the
  hallucination probes, there is no measured second option in this project — and
  nothing in this document says anything about its answer quality (§5).
- Local inference runs with **0.18 GB spare**, so it is sensitive to anything else
  using the GPU.

---

## 5. What is verified, and what is not

### Verified by running on this laptop

- CUDA, bf16 and a real 4-bit bitsandbytes matmul all work.
- MedGemma-4B loads in 4-bit and generates (forced to GPU 0); 3.14 GB peak.
- MedGemma-4B LoRA training exceeds VRAM and spills, at every setting tried.
- Qwen2-VL-2B loads, generates and LoRA-trains inside real VRAM (§2.1) —
  reference only, not used in the project.
- Peak VRAM and step times in the tables above.
- GPU stayed at or below 69 °C.
- `datasets`/`pandas`/`pyarrow.compute` are blocked by Smart App Control.
- `set_per_process_memory_fraction` does not prevent the WDDM sysmem spill.

### Not verified — open questions

- **Nothing about model quality.** No clinical accuracy, no hallucination
  measurement, nothing about whether MedGemma-4B actually abstains or confabulates
  on absent-object probes. The only generation tested was one blank grey image, and
  "No" proves only that the forward pass completes. **This is the main risk in the
  §4 decision**: the project is now committed to MedGemma-4B with no measured
  fallback, on evidence that covers memory and speed only.
- **Whether the cloud path works at all.** Kaggle/Colab was never touched: no
  training run, no adapter saved, no adapter loaded back onto the laptop for
  inference. The round trip that §4 depends on is untested end to end.
- **Real images.** All tests used a flat grey square. For MedGemma this matters
  less for token count — the vision tower is fixed at 896×896, so real X-rays
  should still give 281 tokens — but it is unverified, and local inference has
  only 0.18 GB of headroom to absorb any surprise.
- **Real training targets.** The dummy answer was "No." (a few tokens). Longer
  supervision will lengthen sequences and raise memory above these numbers.
- **Batch size > 1 and gradient accumulation** were never tested; only batch size 1.
- **Sustained training.** Longest run was ~7 minutes and 3 steps. Throughput,
  thermals and stability over thousands of steps are unknown — and on cloud
  hardware none of the timings here apply at all.
- **Which modules LoRA actually attached to.** `target_modules` matches by name
  suffix, so it may have wrapped vision-tower attention as well as the language
  model (MedGemma reported 5.95 M trainable params). Note Gemma 3's SigLIP tower
  uses `out_proj`, not `o_proj`, so the match is probably partial. This should be
  pinned down deliberately before real training.
- **Local inference alongside the Streamlit app.** Untested, and the 0.18 GB of
  free VRAM makes it doubtful. Needs checking before the app is wired to a real
  local model.
- **8-bit / paged optimisers** were not used; plain AdamW. `PagedAdamW8bit` is
  available and would save a little more.
- **The data pipeline**, blocked by the `datasets` issue above.

---

## 6. Reproducing

The project model:

```bash
python model/scripts/check_env.py
python model/scripts/test_load.py                    # MedGemma-4B inference
python model/scripts/test_train_memory.py            # MedGemma-4B training: spills, ~148 s/step
```

The §2.1 reference measurements, kept only for reproducibility of this document
(Qwen2-VL-2B is not used in the project):

```bash
python model/scripts/test_load.py Qwen/Qwen2-VL-2B-Instruct
python model/scripts/test_train_memory.py Qwen/Qwen2-VL-2B-Instruct --image-size 448 --lora-r 8
```

MedGemma-4B is gated: accept its licence on Hugging Face and run
`huggingface-cli login` once first. Its weights are 8.01 GB on disk.
