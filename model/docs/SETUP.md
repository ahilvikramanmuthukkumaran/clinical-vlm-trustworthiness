# Setup — model side

How to install the model dependencies, call `predict()` with and without a GPU,
and re-run the evaluation. For what the model *is*, see `MODEL.md`; for the API,
`INTERFACE.md`.

---

## 1. Install

The model dependencies are kept separate from the Streamlit app's
`requirements.txt`. Use a virtual environment in the repo root (`.venv` is
git-ignored).

```bash
python -m venv .venv
```

Then, on Windows:

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
```

**Install PyTorch with CUDA first**, from the index matching your driver — not
from PyPI, which gives a CPU-only build. Check your driver with `nvidia-smi`.
The cu126 index was used here:

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then the rest:

```bash
.venv/Scripts/python.exe -m pip install -r model/requirements-model.txt
```

Verify the stack:

```bash
.venv/Scripts/python.exe model/scripts/check_env.py
```

You want `CUDA available: True`. Python 3.13 was used; `bitsandbytes` 0.50.2
works on Windows.

### MedGemma is gated

Accept the licence for `google/medgemma-4b-it` on Hugging Face, then log in
**yourself**:

```bash
huggingface-cli login
```

Do not put a token in any file in this repo. The weights are 8.01 GB on first
download.

---

## 2. Running `predict()`

### Stub mode — no GPU needed

If `torch` is missing or no CUDA device is usable, `predict()` returns fixed
stub answers with **`is_stub=True`** instead of failing. This is what lets the
Streamlit app and the evaluation harness import and run anywhere.

```python
from model.src.predict import predict

r = predict("data/images/428d07ef3b0f.jpg", "are regions of the brain infarcted?")
print(r.answer, r.confidence, r.is_stub)   # 'yes' 0.5 True
```

`torch` and `transformers` are imported lazily, so importing this module does
not require them.

> **Evaluation code must filter on `is_stub=False`.** Otherwise a GPU-less run
> silently scores placeholder answers as if they were predictions.

### Real mode — needs a CUDA GPU

```python
from model.src.predict import predict

# baseline: plain prompt, no adapter, never abstains
r = predict(image, question, mode="baseline")

# the final system: adapter + training prompt + threshold 0.8
r = predict(image, question, mode="mitigated",
            adapter_path="model/checkpoints/medgemma_vqarad_lora")
```

`mode="mitigated"` defaults to `prompt_style="train"`. `prompt_style="strict"`
is selectable but abstains on ~80% of questions — see `MODEL.md` §5.

Smoke test both modes on one image:

```bash
.venv/Scripts/python.exe model/scripts/smoke_predict.py
```

Tunables live in `model/config.json` (threshold, `min_yes_no_mass`,
`max_new_tokens`, default prompt style) and need no code change.

---

## 3. GPU and memory requirements

| Task | Requirement |
|---|---|
| **4-bit inference** | **~3.14 GB peak VRAM.** Fits a 4 GB card, but only just |
| **LoRA training** | **a ~15 GB GPU** (a Kaggle T4 at 16 GB is enough) |
| Model weights on disk | 8.01 GB (base) + 34 MB (adapter) |
| System RAM | 16 GB was sufficient |

Measured peaks are in `FIT_TEST.md`. Two practical consequences on a 4 GB card:

- **`device_map={"": 0}` is mandatory.** `device_map="auto"` fails outright —
  accelerate reserves activation headroom, decides the model will not fit and
  raises `ValueError: Some modules are dispatched on the CPU or the disk`.
  `predict()` already forces GPU 0.
- **Only ~0.18 GB of VRAM is free after loading**, because the desktop
  compositor already holds ~0.77 GB. Close other GPU applications. Whether the
  model can share the GPU with the Streamlit app is untested and doubtful.

**Training does not fit a 4 GB card.** It allocates 6.09 GB, which Windows
silently satisfies from shared system RAM, so it *runs* at ~148 s per step
(~41 h per 1000 steps) rather than failing. Lowering the image size or LoRA rank
does not help: Gemma 3's vision tower is fixed at 896×896 and the loss over a
262k-token vocabulary is a fixed cost.

### Precision

`float16` is never used. On a T4 it yields NaN logits and empty output.
`predict()` selects `bfloat16` only where bf16 is native (compute capability
≥ 8.0) and `float32` otherwise. Do not override this.

---

## 4. Known blocker on the author's laptop

**`torch` is blocked by Windows Smart App Control on the development machine.**
`VerifiedAndReputablePolicyState = 1` (enforced), and `torch_python.dll` is
unsigned, so:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading ...\.venv\Lib\site-packages\torch\lib\torch_python.dll
```

No local inference or training is possible while this holds. It also blocks
`pyarrow._compute`, and therefore `pandas` 3.x and `datasets`.

Two ways out, neither applied:

- Run on a cloud GPU instead — what the project does (§5 below).
- Disable Smart App Control. **This is irreversible without reinstalling
  Windows**, so it is not recommended just for this.

If you hit it, note that `predict()` degrades to the stub path rather than
crashing, so `is_stub=False` is the only safe signal that a real model ran.

---

## 5. Re-running the evaluation on Kaggle

`model/notebooks/medgemma_eval.ipynb` is self-contained: it writes out
`predict.py` and `config.json`, downloads the adapter and data, runs three
configurations over both splits and uploads the results.

1. Upload the notebook to Kaggle.
2. **Settings → Accelerator → GPU T4**; **Internet → On**.
3. Add two Kaggle secrets:
   - `HF_TOKEN` — read access, used for every download.
   - `HF_WRITE_TOKEN` — write access, used only by the final upload cell.
4. Accept the MedGemma licence on the same Hugging Face account.
5. **Save Version → Save & Run All (Commit)**.

It runs `baseline`, `adapter_train` and `adapter_strict` over `val.json` and
then `test.json` — about 1671 generations. Rows are appended and fsynced one at
a time to `results/<split>_<config>.jsonl`, so a re-run resumes rather than
recomputing, and previous results are pulled back from the Hugging Face repo
first. **No abstention threshold is applied in the notebook**
(`threshold_applied: 0.0`); thresholds are swept afterwards from `confidence`,
`raw_text` and `gold_answer`.

The notebook aborts after the first generation of each configuration if the
logits are not finite or the output text is empty, so a bad dtype cannot quietly
produce a file of blank answers.

If you change `model/src/predict.py` or `model/config.json`, regenerate the
notebook rather than hand-editing it:

```bash
.venv/Scripts/python.exe model/scripts/build_eval_notebook.py
```
