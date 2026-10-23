
"""
Evaluation framework for:
Enhancing Clinical Trustworthiness:
Reducing Object and Text Hallucinations in Medical VLMs.

Note:
- Object support must come from verified annotations.
- Clinical claim support must be manually or reliably annotated.
- This module calculates metrics; it does not automatically
  determine whether a medical claim is clinically correct.
"""

from typing import Any


def _normalize(text: str) -> str:
    """Normalize labels for basic, case-insensitive comparison."""
    return " ".join(text.strip().lower().split())


def object_hallucination_metrics(
    predicted_objects: list[str],
    verified_objects: list[str],
) -> dict[str, Any]:
    """
    Compare predicted object labels against verified image annotations.

    Exact normalized label matching is used in this initial version.
    Synonyms and clinically equivalent labels require a mapping later.
    """
    predicted = [_normalize(obj) for obj in predicted_objects]
    verified = {_normalize(obj) for obj in verified_objects}

    supported = [obj for obj in predicted if obj in verified]
    unsupported = [obj for obj in predicted if obj not in verified]

    total = len(predicted)

    return {
        "predicted_object_count": total,
        "supported_object_count": len(supported),
        "hallucinated_object_count": len(unsupported),
        "object_hallucination_rate": (
            len(unsupported) / total if total else 0.0
        ),
        "unsupported_objects": unsupported,
    }


def clinical_text_hallucination_metrics(
    clinical_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate unsupported-claim rate from annotated clinical claims.

    Each claim must contain:
        {"text": "Claim text", "supported": True or False}

    The supported label must come from verified reference evidence
    and an appropriate clinical annotation procedure.
    """
    for claim in clinical_claims:
        if not isinstance(claim.get("text"), str):
            raise ValueError("Every claim must contain text as a string.")

        if not isinstance(claim.get("supported"), bool):
            raise ValueError(
                "Every claim must have a Boolean 'supported' label."
            )

    total = len(clinical_claims)
    unsupported = [
        claim["text"]
        for claim in clinical_claims
        if not claim["supported"]
    ]

    return {
        "clinical_claim_count": total,
        "unsupported_claim_count": len(unsupported),
        "clinical_hallucination_rate": (
            len(unsupported) / total if total else 0.0
        ),
        "unsupported_claims": unsupported,
    }


def evaluate_example(
    predicted_objects: list[str],
    verified_objects: list[str],
    clinical_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one image-question-model output."""
    return {
        "object_metrics": object_hallucination_metrics(
            predicted_objects, verified_objects
        ),
        "clinical_text_metrics": clinical_text_hallucination_metrics(
            clinical_claims
        ),
    }


def compare_models(
    baseline_results: list[dict[str, Any]],
    mitigated_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare paired evaluation results.

    Each result must contain:
        object_hallucination_rate
        clinical_hallucination_rate

    Both lists must use the same examples in the same order.
    """
    if len(baseline_results) != len(mitigated_results):
        raise ValueError(
            "Baseline and mitigated results must have equal lengths."
        )

    if not baseline_results:
        raise ValueError("At least one evaluation example is required.")

    metrics = [
        "object_hallucination_rate",
        "clinical_hallucination_rate",
    ]

    comparison = {}

    for metric in metrics:
        baseline_avg = sum(
            result[metric] for result in baseline_results
        ) / len(baseline_results)

        mitigated_avg = sum(
            result[metric] for result in mitigated_results
        ) / len(mitigated_results)

        comparison[metric] = {
            "baseline_mean": baseline_avg,
            "mitigated_mean": mitigated_avg,
            "absolute_reduction": baseline_avg - mitigated_avg,
        }

    return comparison
