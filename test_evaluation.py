
import pytest

from evaluation import (
    object_hallucination_metrics,
    clinical_text_hallucination_metrics,
    evaluate_example,
    compare_models,
)


def test_object_hallucination_rate():
    result = object_hallucination_metrics(
        ["lung", "cat"],
        ["lung"],
    )

    assert result["supported_object_count"] == 1
    assert result["hallucinated_object_count"] == 1
    assert result["object_hallucination_rate"] == 0.5


def test_all_objects_supported():
    result = object_hallucination_metrics(
        ["lung", "heart"],
        ["lung", "heart"],
    )

    assert result["object_hallucination_rate"] == 0.0


def test_no_predicted_objects():
    result = object_hallucination_metrics([], ["lung"])

    assert result["predicted_object_count"] == 0
    assert result["object_hallucination_rate"] == 0.0


def test_clinical_hallucination_rate():
    claims = [
        {"text": "Claim A", "supported": True},
        {"text": "Claim B", "supported": False},
    ]

    result = clinical_text_hallucination_metrics(claims)

    assert result["clinical_claim_count"] == 2
    assert result["unsupported_claim_count"] == 1
    assert result["clinical_hallucination_rate"] == 0.5


def test_invalid_clinical_annotation():
    claims = [
        {"text": "Claim A", "supported": "yes"},
    ]

    with pytest.raises(ValueError):
        clinical_text_hallucination_metrics(claims)


def test_evaluate_example():
    result = evaluate_example(
        predicted_objects=["lung", "cat"],
        verified_objects=["lung"],
        clinical_claims=[
            {"text": "Claim A", "supported": True},
            {"text": "Claim B", "supported": False},
        ],
    )

    assert result["object_metrics"]["object_hallucination_rate"] == 0.5
    assert (
        result["clinical_text_metrics"]["clinical_hallucination_rate"]
        == 0.5
    )


def test_compare_baseline_and_mitigated():
    baseline = [
        {
            "object_hallucination_rate": 0.5,
            "clinical_hallucination_rate": 0.4,
        }
    ]

    mitigated = [
        {
            "object_hallucination_rate": 0.2,
            "clinical_hallucination_rate": 0.3,
        }
    ]

    result = compare_models(baseline, mitigated)

    assert result["object_hallucination_rate"]["absolute_reduction"] == pytest.approx(0.3)
    assert result["clinical_hallucination_rate"]["absolute_reduction"] == pytest.approx(0.1)


def test_compare_requires_equal_sample_counts():
    with pytest.raises(ValueError):
        compare_models(
            [{"object_hallucination_rate": 0.5,
              "clinical_hallucination_rate": 0.4}],
            [],
        )
