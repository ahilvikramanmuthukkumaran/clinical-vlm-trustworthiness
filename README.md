# Clinical VLM Trustworthiness

**Enhancing Clinical Trustworthiness: Reducing Object and Text Hallucinations in Medical Vision-Language Models**

A research prototype that investigates medical vision-language model (VLM) responses using a baseline model and a LoRA-adapted MedGemma model, with an emphasis on answer quality, confidence, and abstention behavior.

> **Disclaimer:** This is an academic research prototype, not a clinical decision-support system. Its predictions must not be used for diagnosis or treatment.

## Project Overview

Medical VLMs can produce answers that are incomplete, overly broad, or inconsistent with the reference answer. This project compares baseline MedGemma responses with responses from a model adapted using LoRA fine-tuning on medical visual question-answering data.

The project includes:
- A Streamlit interface for uploading medical images and entering clinical questions.
- Input validation for image files and questions.
- Baseline and mitigated prediction modes.
- A saved evaluation dataset containing baseline and adapter predictions.
- A script for comparing predictions against reference answers.

## Methodology

### Baseline

The base MedGemma model answers medical visual question-answering prompts without the trained LoRA adapter.

### Mitigated model

The trained LoRA adapter is applied to MedGemma using the training-compatible prompt style. The project also contains results from a stricter prompt configuration designed to encourage answers grounded in visible evidence and appropriate abstention.

### Evaluation

The evaluation script compares saved baseline and adapter predictions using:
- **Exact-match rate:** the proportion of normalized predictions identical to their reference answers.
- **Mean token F1:** average lexical overlap between predictions and reference answers.
- **Abstention rate:** the proportion of predictions identified as abstentions.
- **Mean reported confidence:** the average confidence value recorded in the prediction files.
- **Mean generation time:** the average recorded generation duration.

These metrics provide a reproducible lexical comparison. They do not, by themselves, establish clinical correctness or a clinically validated hallucination rate.

## Evaluation Results

The following results were computed from the saved test predictions for **337 matched test examples**.

| Metric | Baseline | LoRA adapter |
|---|---:|---:|
| Exact-match rate | 0.00% (0/337) | 49.85% (168/337) |
| Mean token F1 | 0.0618 | 0.5692 |
| Abstention rate | 0.00% | 0.00% |
| Mean reported confidence | 0.8729 | 0.7852 |
| Mean generation time | 6.3528 s | 3.3491 s |

### Interpretation

On this saved test set, the adapter achieved a higher exact-match rate and mean token F1 than the baseline. Its recorded mean generation time was also lower.

The baseline's zero exact-match rate does not mean every baseline response was clinically wrong: exact string matching does not recognize semantically equivalent answers. Token F1 provides an additional lexical-overlap measure but is not a substitute for expert clinical evaluation.

The lower average reported confidence of the adapter should not be interpreted as improved calibration without a dedicated calibration analysis. Neither configuration recorded abstentions in these saved test results, so these results do not demonstrate a reduction in abstention-related risk or prove a reduction in hallucination rate.

## Repository Structure

```text
clinical-vlm-trustworthiness/

 # Run
 python -m pytest -q --ignore=model/scripts/test_load.py
git status --short

