# Model Card — MedGemma-4B + LoRA, abstention-gated

Project: *Enhancing Clinical Trustworthiness: Reducing Object and Text
Hallucinations in Medical VLMs*. This file documents the final model-development
deliverable: what was trained, how it is evaluated, and what it achieves.

Related: `INTERFACE.md` (the API the app and evaluation call), `FIT_TEST.md`
(hardware measurements), `SETUP.md` (how to install and run).

---

## 1. The final system

| | |
|---|---|
| Base model | `google/medgemma-4b-it` (gated on Hugging Face) |
| Quantisation | 4-bit NF4, double quantisation |
| Adapter | LoRA, committed at `model/checkpoints/medgemma_vqarad_lora/` |
| Prompt | the exact training template (`prompt_style="train"`) |
| Abstention | confidence < **0.8** → abstain |
| Compute dtype | float32 on a T4; bfloat16 only where bf16 is native (capability ≥ 8.0) |

Called as:

```python
from model.src.predict import predict

result = predict(image, question, mode="mitigated",
                 adapter_path="model/checkpoints/medgemma_vqarad_lora")
```

`mode="baseline"` is the comparison system: same base model, plain prompt, **no
adapter**, and it never abstains.

---

## 2. Training recipe

| Setting | Value |
|---|---|
| Method | QLoRA — 4-bit NF4 frozen base + trainable LoRA adapter |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` of the **language model** only (regex `.*language_model.*\.(q_proj\|k_proj\|v_proj\|o_proj)`) |
| Loss | **answer-only** — the prompt and image tokens are masked out of the loss |
| Prompt template | `Answer the clinical question about this image briefly. {q}` |
| Optimiser steps | **22** |
| Effective batch | 16 (micro-batch 4 × gradient accumulation 4) |
| Samples seen | 352 |
| Epochs | **0.21** of the 1687-example training split |
| Compute dtype | **float32** |
| Hardware | Kaggle T4 (compute capability 7.5), ~184–216 s per optimiser step, ~79 min total |

**float32 is not a stylistic choice.** On a T4 this model in fp16 produces NaN
logits and empty output (only token id 0) while still completing without
raising — a run looks successful and writes blank answers. bf16 is emulated on
Turing and is slower than float32 here. Numbers are in `FIT_TEST.md`.

A caution recorded there and worth repeating: in a short test the **fp16
training loss looked finite even while generation was already invalid**,
probably because `prepare_model_for_kbit_training` casts several layers to
float32 (unverified). So a healthy-looking loss does not validate the dtype —
only generation does.

### What 22 steps means

0.21 epoch is a very short run. The adapter has seen 352 of 1687 training
examples once. Any claim about what the adapter "learned" should be read with
that in mind; the effect measured in §5 is real but small and cheaply obtained,
and was not tuned.

---

## 3. Data

VQA-RAD derived, **split by image** so no image appears in two splits:

| Split | Examples | Images | File |
|---|---|---|---|
| train | 1687 | 236 | `data/splits/train.json` |
| val | 220 | 31 | `data/splits/val.json` |
| test | 337 | 47 | `data/splits/test.json` |

Split seed: **42**. Images live at `data/images/` (314 JPEGs), and
`image_path` values (`images/<id>.jpg`) resolve relative to `data/`.

Per-example fields are documented in `INTERFACE.md` §3. Two matter here:

- `question_type` — `closed` (yes/no) or `open`.
- `category` — `normal`, or **`closed_no`**: a closed question whose
  ground-truth answer is "no". This is the project's **absent-finding probe**:
  asking about something that is not there, where a hallucinating model answers
  "yes".

Test composition: 165 closed / 172 open; 79 of the closed are `closed_no`.

---

## 4. Confidence and abstention

Confidence comes from the generation's token probabilities under greedy
decoding (`output_scores=True`). Two formulas, chosen per question:

- **Closed questions** — the probability mass of *yes* vs *no* spellings on the
  **first generated token**, reported as the winning side's share. 0.5 is a coin
  flip, 1.0 means all the yes/no mass is on one side.
- **Open questions, and closed questions the model does not answer yes/no** —
  the **mean probability of the generated tokens**.

The yes/no formula is only trusted when yes and no together hold at least
`min_yes_no_mass` (**0.10**) of the first-token distribution. The ratio is
scale-free, so without that floor a hedged prose answer carrying ~3% of the mass
— or a refusal carrying ~0.05% — reports a confident-looking 0.89 or 0.97.
Below the floor, confidence falls back to the mean token probability.

**Abstention rule:** in `mitigated` mode, if `confidence < 0.8`, the answer is
replaced with `"cannot determine from this image"` and `abstained=True`. The
unredacted generation is always kept in `raw_text`, so thresholds can be
re-swept offline without re-running the model.

Confidence is **token-level, not calibrated correctness** — no calibration was
measured. When the model abstains, the reported confidence describes the
refusal text, so read `abstained`, not `confidence`, to tell whether an answer
was given.

---

## 5. Results (test split, 337 examples)

| Metric | Baseline | Mitigated (adapter + τ=0.8) |
|---|---|---|
| Closed accuracy | 65.5% | 64.2% |
| False-yes rate (on `closed_no`) | 29.1% | **19.0%** |
| Committed error | 47.8% | **10.1%** |
| Abstention rate | 0% (never abstains) | 48.4% |
| Overall accuracy | — | 41.5% |

The headline claim: **committed error falls from 47.8% to 10.1%**, and the
false-yes rate on the absent-finding probe falls from 29.1% to 19.0%, at the
cost of abstaining on 48.4% of questions and ~1.3 points of closed accuracy.
For a clinical decision-support setting, declining to answer is a far cheaper
failure than a confident wrong answer, which is the premise this result
supports.

### The `strict` prompt was rejected

A third configuration used a strict visible-evidence prompt instructing the
model to say "cannot determine from this image" when unsure. It **abstains on
about 80% of questions** (measured 78.6% on val, 70.3% on test at τ=0.8), which
makes the system useless regardless of its error rate on the remainder. It is
**not** the mitigated default. It stays selectable as `prompt_style="strict"`
for ablation and error analysis only.

### Provenance of these numbers

The six per-example result files are committed at `results/`:
`{val,test}_{baseline,adapter_train,adapter_strict}.jsonl`, 220 rows for val
and 337 for test, each row carrying `confidence`, `raw_text` and `gold_answer`
with **no threshold applied** (`threshold_applied: 0.0`). Thresholds are a
downstream analysis choice, applied after the fact.

The aggregate metrics in the table above were computed by the evaluation side.
Re-deriving them from `results/` independently:

- **Abstention rate 48.4%** — reproduced exactly (`confidence < 0.8` over all
  337 test rows).
- **False-yes 19.0%** — reproduced exactly (committed `closed_no` answers
  parsed as "yes", over all 79 `closed_no` rows).
- Closed accuracy, committed error and overall accuracy **could not be
  reproduced independently**, because they depend on the evaluation side's
  yes/no parser and answer-matching rules, which live outside `model/`. A
  deliberately simple heuristic gives different values (e.g. 49.1% rather than
  65.5% baseline closed accuracy), which reflects the weakness of that
  heuristic, not a disagreement with the reported figures. Treat the table as
  the evaluation side's output, with `results/` as the raw material.

---

## 6. Caveats

These bound what the results above can support:

- **The test set is small: 337 questions over only 47 images.** Differences of
  a point or two are not meaningful, and per-image idiosyncrasies are not
  averaged out. The closed accuracy difference (65.5% vs 64.2%) is well inside
  that noise; the committed-error difference is not.
- **`closed_no` is a proxy for hallucination, not a measurement of it.** It
  captures "a closed question whose true answer is no", which is a reasonable
  stand-in for asking about an absent finding, but it is not a curated
  absent-object probe and it says nothing about hallucinated *text* detail in
  open answers.
- **Baseline yes/no answers are parsed by a heuristic.** The baseline was never
  instructed to answer yes/no, and often replies in prose ("Based on the image
  provided, it is difficult to…"). Any parser over that is lossy, and the
  baseline's closed accuracy and false-yes rate inherit that error. The
  mitigated system, prompted with the training template, answers far more
  tersely and is less exposed to this.
- **The adapter is barely trained** — 22 optimiser steps, 0.21 epoch, no
  hyperparameter search, no seed replication. Nothing here separates the
  adapter's contribution from the prompt's.
- **No calibration.** τ=0.8 was selected against the val split, not derived
  from a calibration curve.
- **Single-image, single-question** inference only.
- **Answers are capped at 48 new tokens**, so a long answer can end
  mid-sentence.
