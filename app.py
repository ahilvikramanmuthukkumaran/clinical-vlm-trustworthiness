
import json
import importlib.util
import sys
from pathlib import Path
from html import escape

import streamlit as st
from PIL import Image, UnidentifiedImageError

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_SRC = MODEL_DIR / "src"
ADAPTER = MODEL_DIR / "checkpoints" / "medgemma_vqarad_lora"

RESULTS_DIR = ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"
REPORT_PATH = RESULTS_DIR / "evaluation_summary.md"
PREDICT_PATH = MODEL_SRC / "predict.py"

for folder in (MODEL_DIR, MODEL_SRC):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

# ============================================================
# IMPORT MODEL
# ============================================================
predict = None
MODEL_IMPORT_ERROR = None

try:
    if not PREDICT_PATH.is_file():
        raise FileNotFoundError(f"Prediction module not found: {PREDICT_PATH}")

    spec = importlib.util.spec_from_file_location(
        "clinical_vlm_predict", PREDICT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load model/src/predict.py")

    predict_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = predict_module
    spec.loader.exec_module(predict_module)
    predict = predict_module.predict

except Exception as exc:
    MODEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from validation import validate_question
    VALIDATION_IMPORT_ERROR = None
except Exception as exc:
    validate_question = None
    VALIDATION_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Clinical VLM Trustworthiness",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.28);
        margin-bottom: 1.5rem;
        background: linear-gradient(
            120deg,
            rgba(45,105,160,.18),
            rgba(65,155,135,.10)
        );
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .78rem;
        opacity: .78;
        margin-bottom: .65rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 780;
        line-height: 1.15;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: .84;
        margin-top: .85rem;
        max-width: 900px;
        line-height: 1.6;
    }
    .section-note {
        opacity: .78;
        font-size: .95rem;
        margin-bottom: 1rem;
    }
    .answer-panel {
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.3);
        margin: .7rem 0 1rem 0;
    }
    .answer-label {
        text-transform: uppercase;
        font-size: .75rem;
        letter-spacing: .12em;
        opacity: .7;
        margin-bottom: .65rem;
    }
    .answer-text {
        font-size: 1.35rem;
        line-height: 1.65;
        font-weight: 550;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        padding: .85rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">NLP Research Prototype</div>
      <div class="hero-title">Clinical VLM Trustworthiness</div>
      <div class="hero-subtitle">
        Evaluating answer consistency, confidence, and hallucination-related
        risks in medical vision-language models using medical visual
        question-answering data.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project Overview")
    st.write(
        "Compare baseline MedGemma responses with a LoRA-adapted model "
        "and review test-set evaluation results."
    )
    st.divider()
    st.markdown(
        "- Medical image upload\n"
        "- Clinical question validation\n"
        "- Baseline and mitigated modes\n"
        "- Test-set evaluation"
    )
    st.caption("Academic research prototype. Not intended for diagnosis or treatment.")

# ============================================================
# INPUT INTERFACE
# ============================================================
st.header("1. Medical Image Analysis")
st.markdown(
    '<div class="section-note">Upload a medical image, enter a question, and select a model configuration.</div>',
    unsafe_allow_html=True,
)

input_col, preview_col = st.columns([1, 1], gap="large")

with input_col:
    uploaded = st.file_uploader(
        "Upload a medical image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG and JPEG.",
    )

    question = st.text_area(
        "Clinical question",
        placeholder="Example: In which lobe are the lesions?",
        height=130,
    )

    mode_label = st.radio(
        "Prediction mode",
        ["Mitigated", "Baseline"],
        horizontal=True,
    )

    run_prediction = st.button(
        "Get prediction",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# IMAGE VALIDATION AND PREVIEW
# ============================================================
image = None

with preview_col:
    st.subheader("Image Preview")

    if uploaded is None:
        st.info("Your uploaded image will appear here.")
    else:
        try:
            uploaded.seek(0)
            with Image.open(uploaded) as check_image:
                check_image.verify()

            uploaded.seek(0)
            with Image.open(uploaded) as opened_image:
                image = opened_image.convert("RGB")

            st.image(
                image,
                caption="Uploaded medical image",
                use_container_width=True,
            )
            st.success("Image file is valid.")

        except (UnidentifiedImageError, OSError, ValueError):
            image = None
            st.error("Invalid image. Please upload a valid PNG or JPEG file.")

# ============================================================
# PREDICTION AND EVALUATION ARE TRIGGERED BY THE BUTTON
# ============================================================
if run_prediction:
    # ---------------- Validate inputs ----------------
    if image is None:
        st.error("Please upload a valid medical image.")
        st.stop()

    if validate_question is None:
        st.error("Question validator could not be imported.")
        st.code(VALIDATION_IMPORT_ERROR or "Unknown import error")
        st.stop()

    valid, message = validate_question(question)

    if not valid:
        st.error(message)
        st.stop()

    if predict is None:
        st.error("Model prediction module could not be imported.")
        st.code(MODEL_IMPORT_ERROR or "Unknown import error")
        st.stop()

    mode = "mitigated" if mode_label == "Mitigated" else "baseline"

    if mode == "mitigated" and not ADAPTER.is_dir():
        st.error(f"Trained adapter folder not found: {ADAPTER}")
        st.stop()

    # ---------------- Run prediction ----------------
    try:
        with st.spinner(
            "Generating prediction. First-time model loading may take a while..."
        ):
            prediction_args = {}

            if mode == "mitigated":
                prediction_args["adapter_path"] = str(ADAPTER)

            result = predict(
                image=image,
                question=question.strip(),
                mode=mode,
                **prediction_args,
            )

        st.divider()
        st.header("2. Prediction Results")

        if result.is_stub:
            st.warning(
                "Stub output: the real model did not run. This is a placeholder, "
                "not a genuine model prediction."
            )
        elif result.abstained:
            st.warning("The model abstained from answering.")
        else:
            st.success("A model response was returned.")

        safe_answer = escape(str(result.answer))

        st.markdown(
            f"""
            <div class="answer-panel">
              <div class="answer-label">Model Answer</div>
              <div class="answer-text">{safe_answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Reported Confidence", f"{result.confidence:.1%}")
        col2.metric("Prediction Mode", result.mode.capitalize())
        col3.metric(
            "Abstained",
            "Yes" if result.abstained else "No",
        )

        if result.is_stub:
            st.caption(
                "The answer and confidence are placeholders because live "
                "inference did not run."
            )

        with st.expander("Technical Details"):
            st.write("**Stub output:**", "Yes" if result.is_stub else "No")
            st.write(
                "**Adapter:**",
                str(ADAPTER) if mode == "mitigated" else "None (baseline)",
            )
            st.write("**Raw model output**")
            st.code(result.raw_text or "(empty)")

    except Exception as exc:
        st.error("Prediction failed.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.info(
            "Check model dependencies, model access, hardware, adapter files, "
            "and available memory."
        )

    # ========================================================
    # EVALUATION DASHBOARD — ONLY SHOWN AFTER BUTTON CLICK
    # ========================================================
    st.divider()
    st.header("3. Model Evaluation Dashboard")
    st.markdown(
        '<div class="section-note">Comparison of previously saved baseline and LoRA-adapter test predictions.</div>',
        unsafe_allow_html=True,
    )

    if SUMMARY_PATH.exists():
        try:
            with SUMMARY_PATH.open("r", encoding="utf-8") as file:
                summary = json.load(file)

            baseline = summary["baseline"]
            adapter = summary["adapter_train"]
            paired = summary["paired_comparison"]
            alignment = summary["dataset_alignment"]

            exact_delta = paired.get(
                "adapter_minus_baseline_exact_match_rate"
            )

            top1, top2, top3 = st.columns(3)

            top1.metric(
                "Matched Test Examples",
                alignment.get("matched_unique_ids", "N/A"),
            )

            top2.metric(
                "Baseline Exact Match",
                (
                    f"{baseline['exact_match_rate']:.2%}"
                    if baseline.get("exact_match_rate") is not None
                    else "N/A"
                ),
            )

            top3.metric(
                "Adapter Exact Match",
                (
                    f"{adapter['exact_match_rate']:.2%}"
                    if adapter.get("exact_match_rate") is not None
                    else "N/A"
                ),
                delta=(
                    f"{exact_delta:+.2%}"
                    if exact_delta is not None
                    else None
                ),
            )

            baseline_col, adapter_col = st.columns(2, gap="large")

            with baseline_col:
                st.subheader("Baseline Model")

                b1, b2 = st.columns(2)
                b1.metric(
                    "Exact-Match Rate",
                    (
                        f"{baseline['exact_match_rate']:.2%}"
                        if baseline.get("exact_match_rate") is not None
                        else "N/A"
                    ),
                )
                b2.metric(
                    "Mean Token F1",
                    (
                        f"{baseline['mean_token_f1']:.4f}"
                        if baseline.get("mean_token_f1") is not None
                        else "N/A"
                    ),
                )

                b1, b2 = st.columns(2)
                b1.metric(
                    "Mean Reported Confidence",
                    (
                        f"{baseline['mean_reported_confidence']:.2%}"
                        if baseline.get("mean_reported_confidence") is not None
                        else "N/A"
                    ),
                )
                b2.metric(
                    "Mean Generation Time",
                    (
                        f"{baseline['mean_generation_seconds']:.2f} s"
                        if baseline.get("mean_generation_seconds") is not None
                        else "N/A"
                    ),
                )

            with adapter_col:
                st.subheader("Mitigated Model (LoRA)")

                a1, a2 = st.columns(2)
                a1.metric(
                    "Exact-Match Rate",
                    (
                        f"{adapter['exact_match_rate']:.2%}"
                        if adapter.get("exact_match_rate") is not None
                        else "N/A"
                    ),
                    delta=(
                        f"{exact_delta:+.2%}"
                        if exact_delta is not None
                        else None
                    ),
                )
                a2.metric(
                    "Mean Token F1",
                    (
                        f"{adapter['mean_token_f1']:.4f}"
                        if adapter.get("mean_token_f1") is not None
                        else "N/A"
                    ),
                )

                a1, a2 = st.columns(2)
                a1.metric(
                    "Mean Reported Confidence",
                    (
                        f"{adapter['mean_reported_confidence']:.2%}"
                        if adapter.get("mean_reported_confidence") is not None
                        else "N/A"
                    ),
                )
                a2.metric(
                    "Mean Generation Time",
                    (
                        f"{adapter['mean_generation_seconds']:.2f} s"
                        if adapter.get("mean_generation_seconds") is not None
                        else "N/A"
                    ),
                )

            st.subheader("Paired Test-Set Comparison")

            pair1, pair2 = st.columns(2)
            pair1.metric(
                "Baseline Mismatch → Adapter Exact Match",
                paired.get("baseline_wrong_adapter_exact", "N/A"),
            )
            pair2.metric(
                "Baseline Exact Match → Adapter Mismatch",
                paired.get("baseline_exact_adapter_not", "N/A"),
            )

            st.caption(
                "Exact match and token F1 measure lexical similarity to reference "
                "answers. They do not independently establish clinical correctness "
                "or a validated hallucination rate. Confidence is not necessarily "
                "calibrated. The dashboard displays saved test predictions and "
                "does not run additional inference."
            )

            with st.expander("View Full Evaluation Report"):
                if REPORT_PATH.exists():
                    st.markdown(
                        REPORT_PATH.read_text(encoding="utf-8")
                    )
                else:
                    st.json(summary)

        except (KeyError, TypeError, ValueError, OSError) as exc:
            st.error("The saved evaluation summary could not be loaded.")
            st.code(f"{type(exc).__name__}: {exc}")

    else:
        st.info(
            "Evaluation summary not found. Run `python evaluate_results.py` "
            "from the project root to generate it."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Clinical VLM Trustworthiness | Academic NLP research prototype. "
    "Model outputs may be incorrect and must not be used as a substitute "
    "for professional medical judgment."
)
