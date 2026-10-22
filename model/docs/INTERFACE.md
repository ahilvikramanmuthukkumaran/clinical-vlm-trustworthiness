# Model Interface Specification (v0.1 draft)

Owner: model development. Consumer: evaluation + Streamlit app.
Status: **final**. The fields below are stable. The model is real
(MedGemma-4B-it, 4-bit) and `mitigated` runs a trained LoRA adapter, committed
at `model/checkpoints/medgemma_vqarad_lora/`. See `model/docs/MODEL.md` for the
training recipe and the evaluated results.

## 1. Function

```python
from model.src.predict import predict

result = predict(image, question, mode="baseline")   # or "mitigated"

# the evaluated system: adapter + training prompt + threshold 0.8
result = predict(image, question, mode="mitigated",
                 adapter_path="model/checkpoints/medgemma_vqarad_lora")
```

| Argument | Type | Notes |
|---|---|---|
| `image` | `str` path, `os.PathLike`, or `PIL.Image.Image` | Converted to RGB; resized internally |
| `question` | `str` | One clinical question, plain text |
| `mode` | `"baseline"` or `"mitigated"` | Same base model; mitigated adds abstention (and takes the adapter) |
| `adapter_path` | `str` or `None` (default `None`) | LoRA adapter directory, loaded with `peft`. `None` = base model only |
| `prompt_style` | `"plain"`, `"strict"`, `"train"` or `None` (default `None`) | Overrides the prompt. `None` uses the mode's default: `baseline` → `"plain"`, `mitigated` → `"train"`. See §2b |

The model is loaded once, lazily, on the first call and cached at module level.
The first call therefore costs ~20-35 s; later calls are ~1.5-3 s. Changing
`adapter_path` between calls triggers a reload.

`torch` and `transformers` are imported lazily, so importing this module does
not require them — the Streamlit app can import `predict` on a machine with no
GPU stack and get stub answers.

## 2. Output (`Prediction`)

| Field | Type | Meaning |
|---|---|---|
| `answer` | `str` | Final answer. If abstained, a fixed "cannot determine" sentence |
| `confidence` | `float` 0-1 | From generated-token probabilities; see below (stub returns 0.5) |
| `abstained` | `bool` | True when the model declines to answer |
| `raw_text` | `str` | Unprocessed generation, for error analysis. **Kept even when abstaining**, so the suppressed answer is still available for error analysis |
| `mode` | `str` | Echo of the mode used |
| `is_stub` | `bool` | `False` for real model output. `True` only on the no-CUDA fallback |

`result.to_dict()` returns a JSON-serialisable dict.

### `is_stub` semantics

- `is_stub=False` — a real MedGemma-4B generation.
- `is_stub=True` — **CUDA was unavailable**, so `predict()` returned the old
  hard-coded stub answers (`"yes"` / `"cannot determine from this image"`, both
  with `confidence=0.5`). This keeps the app and the evaluation harness runnable
  on a GPU-less machine.

Evaluation code must **filter on `is_stub=False`**, or a GPU-less run will
silently score stub answers as if they were model predictions.

### How `confidence` is computed

Greedy decoding with `output_scores=True`; two formulas, chosen per question:

- **Closed (yes/no) questions** — the probability mass of *yes* vs *no* spellings
  on the **first generated token**, reported as the winning side's share. So 0.5
  is a coin flip and 1.0 means all the yes/no mass sits on one side.
- **Open questions, and closed questions the model does not answer yes/no** —
  the **mean probability of the generated tokens**.

Closed questions are detected by a heuristic on the question text (leading
`is/are/does/can/has/…`, or the phrase "yes or no"); `predict()` is not passed
the `question_type` field from the test split.

The yes/no formula is only used when yes and no together hold at least
`min_yes_no_mass` of the first-token distribution (default 0.10). This guard is
load-bearing: the ratio is scale-free, and measured on this model a real "Yes"
carries ~0.97 of the first-token mass while a hedged prose answer carries ~0.03
and a refusal ~0.0005 — where the unguarded ratio reported 0.89 and 0.97. In
mitigated mode that inflation is the dangerous direction, because high
confidence suppresses abstention. Below the floor, confidence falls back to the
mean generated-token probability.

Two caveats for whoever interprets these numbers:

- This is **token-level confidence, not calibrated correctness**. No calibration
  has been measured.
- When the model abstains, the reported confidence describes the *refusal* text,
  so an abstention can carry a high number (e.g. 0.986). Read `abstained`, not
  `confidence`, to tell whether an answer was given.

### Mode behaviour

| | `baseline` | `mitigated` |
|---|---|---|
| Prompt | plain clinical VQA | the exact training template (`prompt_style="train"`) |
| Abstains? | never (`abstained` always `False`) | yes — when confidence < `abstention_threshold`, or when the model itself emits the abstain phrase |
| Confidence | computed | computed |
| LoRA adapter | not used | `model/checkpoints/medgemma_vqarad_lora/`, passed via `adapter_path` |

## 2b. Configuration — `model/config.json`

Runtime settings, tunable without code changes. Missing or malformed file falls
back to these defaults.

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `google/medgemma-4b-it` | Base model |
| `abstention_threshold` | `0.8` | `mitigated` abstains below this confidence. `0.0` disables, `1.0` always abstains. 0.8 is the evaluated operating point |
| `default_mitigated_prompt_style` | `"train"` | prompt `mitigated` uses when the caller passes none. `"strict"` stays selectable per call but must not be the default |
| `max_new_tokens` | `48` | Generation cap. Also hard-capped at 48 in code |
| `min_yes_no_mass` | `0.10` | Minimum first-token yes/no mass before the yes-vs-no ratio is trusted. `0.0` = unguarded ratio |
| `abstain_answer` | `cannot determine from this image` | Exact abstention string, used in both the prompt and the output |

Keys starting with `_` are comments and ignored.

## 3. Test split (shared by both modes)

File: `data/splits/test.json` (list of objects)

| Field | Meaning |
|---|---|
| `id` | unique string |
| `image_path` | relative path to the image |
| `question` | question text |
| `answer` | ground-truth answer |
| `question_type` | `closed` (yes/no) or `open` |
| `category` | `normal` or `closed_no` (the absent-finding probe: a closed question whose ground-truth answer is "no") |

Both baseline and mitigated models MUST be evaluated on exactly this file.

## 4. Artifacts (to be filled in as they exist)

- Base model: **`google/medgemma-4b-it`**, 4-bit nf4, bf16 compute,
  `device_map={"": 0}` (decided in `model/docs/FIT_TEST.md`)
- LoRA adapter: **`model/checkpoints/medgemma_vqarad_lora/`, committed to Git**
  (34 MB). r=16, alpha=32, on the language model's q/k/v/o projections
- Config: `model/config.json`
- Setup: `model/requirements-model.txt` + `model/docs/SETUP.md`
- Smoke test: `model/scripts/smoke_predict.py`

## 5. Known limits

- Single image, single question per call.
- First call loads the model (~20-35 s); later calls are ~1.5-3 s.
- Needs a CUDA GPU for real inference; the stub runs anywhere, with `is_stub=True`.
- **Only ~0.18 GB of VRAM is free** after the model loads on a 4 GB card. Hence
  batch size 1, `max_new_tokens` capped at 48, and the real risk that the model
  cannot share the GPU with the Streamlit app. Untested together so far.
- `mitigated` combines the trained adapter, the training prompt and the
  confidence threshold. The abstention itself is a threshold rule, not learned.
- Confidence is uncalibrated (see §2).
- Long answers are truncated at 48 tokens, so `answer` may end mid-sentence.
