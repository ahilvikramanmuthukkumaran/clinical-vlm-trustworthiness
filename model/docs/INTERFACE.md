# Model Interface Specification (v0.1 draft)

Owner: model development. Consumer: evaluation + Streamlit app.
Status: DRAFT. Fields below are stable; the model behind them is not final yet.

## 1. Function

```python
from model.src.predict import predict

result = predict(image, question, mode="baseline")   # or "mitigated"
```

| Argument | Type | Notes |
|---|---|---|
| `image` | `str` path or `PIL.Image.Image` | RGB; resized internally |
| `question` | `str` | One clinical question, plain text |
| `mode` | `"baseline"` or `"mitigated"` | Same base model; mitigated adds LoRA adapter + abstention |

## 2. Output (`Prediction`)

| Field | Type | Meaning |
|---|---|---|
| `answer` | `str` | Final answer. If abstained, a fixed "cannot determine" sentence |
| `confidence` | `float` 0-1 | Derived from token probabilities (stub returns 0.5) |
| `abstained` | `bool` | True when the model declines to answer |
| `raw_text` | `str` | Unprocessed generation, for error analysis |
| `mode` | `str` | Echo of the mode used |
| `is_stub` | `bool` | True until the real model replaces the stub |

`result.to_dict()` returns a JSON-serialisable dict.

## 3. Test split (shared by both modes)

File: `data/splits/test.json` (list of objects)

| Field | Meaning |
|---|---|
| `id` | unique string |
| `image_path` | relative path to the image |
| `question` | question text |
| `answer` | ground-truth answer |
| `question_type` | `closed` (yes/no) or `open` |
| `category` | `normal` or `absent_object` (probe: the asked-about finding is not in the image) |

Both baseline and mitigated models MUST be evaluated on exactly this file.

## 4. Artifacts (to be filled in as they exist)

- Base model: TBD (MedGemma-4B or Qwen2-VL-2B, decided after the fit test)
- LoRA adapter: `model/checkpoints/<name>/` (not committed to Git; shared via link)
- Setup: `model/requirements-model.txt` + `model/docs/SETUP.md`

## 5. Known limits

- Single image, single question per call.
- First call loads the model (slow); later calls are faster.
- Needs a CUDA GPU for real inference; the stub runs anywhere.
