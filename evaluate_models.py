
"""
Model-independent experiment runner.

IMPORTANT:
All examples and predictions in this file are synthetic.
These results validate the evaluation pipeline only.
They are not real medical VLM experimental findings.
"""

import csv
from pathlib import Path

from evaluation import evaluate_example, compare_models


OUTPUT_DIR = Path("results")


# Synthetic examples for testing the pipeline.
# Replace these with a properly annotated real test set later.
SYNTHETIC_DATASET = [
    {
        "example_id": "SYN001",
        "question": "[SYNTHETIC] What structures are visible?",
        "verified_objects": ["lung", "heart"],
        "baseline_objects": ["lung", "heart", "cat"],
        "mitigated_objects": ["lung", "heart"],
        "baseline_claims": [
            {"text": "The lungs are visible.", "supported": True},
            {"text": "A mass is present.", "supported": False},
        ],
        "mitigated_claims": [
            {"text": "The lungs are visible.", "supported": True},
        ],
    },
    {
        "example_id": "SYN002",
        "question": "[SYNTHETIC] Describe the visible structures.",
        "verified_objects": ["lung"],
        "baseline_objects": ["lung", "tumor"],
        "mitigated_objects": ["lung", "heart"],
        "baseline_claims": [
            {"text": "The lung is visible.", "supported": True},
            {"text": "A tumor is present.", "supported": False},
        ],
        "mitigated_claims": [
            {"text": "The lung is visible.", "supported": True},
            {"text": "The heart is visible.", "supported": False},
        ],
    },
    {
        "example_id": "SYN003",
        "question": "[SYNTHETIC] Identify the visible anatomy.",
        "verified_objects": ["heart", "lung"],
        "baseline_objects": ["heart", "lung"],
        "mitigated_objects": ["heart"],
        "baseline_claims": [
            {"text": "The heart is visible.", "supported": True},
        ],
        "mitigated_claims": [
            {"text": "The heart is visible.", "supported": True},
        ],
    },
    {
        "example_id": "SYN004",
        "question": "[SYNTHETIC] What finding is supported?",
        "verified_objects": ["lung"],
        "baseline_objects": ["lung", "fracture"],
        "mitigated_objects": ["lung"],
        "baseline_claims": [
            {"text": "The lung is visible.", "supported": True},
            {"text": "A fracture is present.", "supported": False},
        ],
        "mitigated_claims": [
            {"text": "The lung is visible.", "supported": True},
        ],
    },
]


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_example_rows = []
    baseline_comparison = []
    mitigated_comparison = []

    for example in SYNTHETIC_DATASET:
        baseline = evaluate_example(
            predicted_objects=example["baseline_objects"],
            verified_objects=example["verified_objects"],
            clinical_claims=example["baseline_claims"],
        )

        mitigated = evaluate_example(
            predicted_objects=example["mitigated_objects"],
            verified_objects=example["verified_objects"],
            clinical_claims=example["mitigated_claims"],
        )

        baseline_row = {
            "example_id": example["example_id"],
            "system": "synthetic_baseline",
            "data_type": "SYNTHETIC_NOT_REAL_MODEL_OUTPUT",
            "object_hallucination_rate":
                baseline["object_metrics"]["object_hallucination_rate"],
            "clinical_hallucination_rate":
                baseline["clinical_text_metrics"][
                    "clinical_hallucination_rate"
                ],
        }

        mitigated_row = {
            "example_id": example["example_id"],
            "system": "synthetic_mitigated",
            "data_type": "SYNTHETIC_NOT_REAL_MODEL_OUTPUT",
            "object_hallucination_rate":
                mitigated["object_metrics"]["object_hallucination_rate"],
            "clinical_hallucination_rate":
                mitigated["clinical_text_metrics"][
                    "clinical_hallucination_rate"
                ],
        }

        per_example_rows.extend([baseline_row, mitigated_row])
        baseline_comparison.append(baseline_row)
        mitigated_comparison.append(mitigated_row)

    # Save individual example results.
    per_example_path = OUTPUT_DIR / "synthetic_per_example.csv"

    with per_example_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=per_example_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(per_example_rows)

    # Compare mean rates across the same examples.
    comparison = compare_models(
        baseline_comparison,
        mitigated_comparison,
    )

    summary_rows = []

    for metric, values in comparison.items():
        summary_rows.append({
            "data_type": "SYNTHETIC_NOT_REAL_MODEL_OUTPUT",
            "metric": metric,
            "baseline_mean": values["baseline_mean"],
            "mitigated_mean": values["mitigated_mean"],
            "absolute_reduction": values["absolute_reduction"],
        })

    summary_path = OUTPUT_DIR / "synthetic_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Synthetic experiment completed successfully.")
    print("WARNING: These are synthetic results, not model findings.")
    print(f"Examples evaluated: {len(SYNTHETIC_DATASET)}")
    print(f"Per-example results: {per_example_path}")
    print(f"Summary comparison: {summary_path}")

    print("\nSynthetic comparison:")
    for metric, values in comparison.items():
        print(f"\n{metric}")
        print(f"  Baseline mean:  {values['baseline_mean']:.3f}")
        print(f"  Mitigated mean: {values['mitigated_mean']:.3f}")
        print(f"  Absolute reduction: {values['absolute_reduction']:.3f}")


if __name__ == "__main__":
    run_experiment()
