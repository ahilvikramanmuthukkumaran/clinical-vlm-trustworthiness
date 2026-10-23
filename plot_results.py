
"""Plot synthetic evaluation results.
These plots are for pipeline testing, not real model findings.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
INPUT_FILE = RESULTS_DIR / "synthetic_summary.csv"
OUTPUT_FILE = RESULTS_DIR / "synthetic_comparison.png"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run evaluate_models.py first."
        )

    with INPUT_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The summary CSV contains no results.")

    metrics = [row["metric"] for row in rows]
    baseline = [float(row["baseline_mean"]) * 100 for row in rows]
    mitigated = [float(row["mitigated_mean"]) * 100 for row in rows]

    positions = list(range(len(metrics)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        [p - width / 2 for p in positions],
        baseline,
        width,
        label="Synthetic baseline",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        mitigated,
        width,
        label="Synthetic mitigated",
    )

    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title(
        "Synthetic Pipeline Test — NOT Real Model Results"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [metric.replace("_", "\n") for metric in metrics]
    )
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.show()

    print(f"Graph saved to: {OUTPUT_FILE}")
    print("Reminder: these are synthetic pipeline-test results.")


if __name__ == "__main__":
    main()
