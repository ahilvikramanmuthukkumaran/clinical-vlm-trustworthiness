"""Public interface of the model module.

Real MedGemma-4B implementation. The signature and the `Prediction` fields are
unchanged from the stub version (see model/docs/INTERFACE.md); only the body of
`predict()` is real now.

The model is loaded once, lazily, on the first real call and cached at module
level. Loading settings are the ones measured to fit this hardware in
model/docs/FIT_TEST.md: 4-bit nf4, bf16 compute, device_map={"": 0}.

If CUDA is unavailable the module falls back to the old stub answers with
is_stub=True, so the Streamlit app and the evaluation harness still import and
run on a machine without a GPU. torch and transformers are imported lazily for
the same reason - importing this module does not require them.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from PIL import Image

Mode = Literal["baseline", "mitigated"]

_MODEL_DIR = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _MODEL_DIR / "config.json"

_DEFAULTS = {
    "model_id": "google/medgemma-4b-it",
    "abstention_threshold": 0.8,
    "max_new_tokens": 48,
    "min_yes_no_mass": 0.10,
    "default_mitigated_prompt_style": "train",
    "abstain_answer": "cannot determine from this image",
}

# Hard ceiling regardless of config: only ~0.18 GB of VRAM is free after load.
_MAX_NEW_TOKENS_CAP = 48

# Questions that start like this are treated as closed (yes/no), which switches
# confidence to the yes-vs-no probability mass on the first generated token.
_CLOSED_START = re.compile(
    r"^\s*(is|are|was|were|does|do|did|can|could|has|have|had|will|would|should|any)\b",
    re.IGNORECASE,
)

_YES_WORDS = ["yes", "Yes", "YES", " yes", " Yes"]
_NO_WORDS = ["no", "No", "NO", " no", " No"]


@dataclass
class Prediction:
    answer: str            # final answer text shown to the user
    confidence: float      # 0.0 - 1.0, from generated-token probabilities
    abstained: bool        # True if the model declined to answer
    raw_text: str          # unprocessed model generation
    mode: str              # "baseline" or "mitigated"
    is_stub: bool = True   # False once the real model is wired in

    def to_dict(self) -> dict:
        return asdict(self)


def load_config() -> dict:
    """Read model/config.json over the defaults. Missing or broken file -> defaults."""
    config = dict(_DEFAULTS)
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            on_disk = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return config
    for key in _DEFAULTS:
        if key in on_disk:
            config[key] = on_disk[key]
    return config


# --------------------------------------------------------------------------
# Model cache
# --------------------------------------------------------------------------

_lock = threading.Lock()
_cache: dict = {"key": None, "model": None, "processor": None}

#: Diagnostics for the most recent predict() generation. Callers use this to
#: detect a broken numeric setup (e.g. fp16 NaN logits on a T4) which cannot be
#: expressed through Prediction's fixed fields. Overwritten on every call.
LAST_GENERATION: dict = {
    "scores_finite": None,     # True/False: every generated step's logits finite
    "non_finite_steps": None,  # how many steps contained NaN/Inf
    "new_tokens": None,        # number of tokens generated
    "text_empty": None,        # True if the decoded text is blank
    "dtype": None,             # compute dtype actually used
    "device_capability": None, # e.g. "8.6"
}


def cuda_available() -> bool:
    """True only if torch is importable and reports a usable CUDA device.

    Importing torch can fail with more than ImportError. On this machine
    Windows Smart App Control blocked torch's unsigned torch_python.dll, which
    surfaces as OSError [WinError 4551]. Any import failure means "no usable
    GPU stack", so callers get the stub rather than a crash.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - ImportError, OSError (blocked DLL), etc.
        return False
    try:
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - a broken driver must not raise here
        return False


#: GPUs from compute capability 8.0 (Ampere) up have bf16 tensor cores.
_NATIVE_BF16_MAJOR = 8


def compute_dtype(torch):
    """bfloat16 only on GPUs with NATIVE bf16, otherwise float32. Never float16.

    float16 is deliberately excluded: on a Kaggle T4 this model produces NaN
    logits and empty output in fp16 (measured by the project owner). bf16
    emulated and float32 both work, and the adapter was trained with float32
    compute, so float32 is the safe fallback.

    The test is compute capability >= 8.0, not torch.cuda.is_bf16_supported():
    recent torch reports True on a T4 because it will emulate bf16, which is
    not what we want to select on. An RTX 3050 is 8.6 and takes the bf16 path;
    a T4 is 7.5 and takes float32.
    """
    try:
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(0)
            if major >= _NATIVE_BF16_MAJOR:
                return torch.bfloat16
    except Exception:  # noqa: BLE001 - never let a probe failure pick fp16
        pass
    return torch.float32


def _build_model(model_id: str, adapter_path: Optional[str]):
    """Load the 4-bit model and processor. Settings verified in FIT_TEST.md."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    dtype = compute_dtype(torch)

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        # Same dtype for the 4-bit compute and, via dtype= below, for every
        # module bitsandbytes leaves unquantised (embeddings, norms, lm_head).
        # Mixing fp32 weights with an fp16 compute dtype is what produced the
        # NaN logits on a T4, so the two must not diverge.
        bnb_4bit_compute_dtype=dtype,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quantisation,
        # device_map="auto" is NOT usable on a 4 GB card: accelerate reserves
        # activation headroom, decides the model will not fit and refuses to
        # place it. Forcing GPU 0 is the configuration measured to work, and is
        # also correct on a single-GPU T4.
        device_map={"": 0},
        dtype=dtype,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


def get_model(adapter_path: Optional[str] = None, model_id: Optional[str] = None):
    """Return the cached (model, processor), loading on first use.

    Reloads if a different adapter_path is requested than the cached one.
    """
    model_id = model_id or load_config()["model_id"]
    key = (model_id, adapter_path or "")
    with _lock:
        if _cache["key"] != key or _cache["model"] is None:
            _cache["model"], _cache["processor"] = _build_model(model_id, adapter_path)
            _cache["key"] = key
        return _cache["model"], _cache["processor"]


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_BASELINE_PROMPT = (
    "You are assisting with reading a medical image. "
    "Answer the following question about the image.\n\nQuestion: {question}"
)

# The "strict" prompt. Selectable, but NOT the mitigated default: evaluated at
# threshold 0.8 it abstains on ~79% of val questions (70% of test), which makes
# the system useless in practice. Kept for ablation and error analysis only.
_MITIGATED_PROMPT = (
    "You are assisting with reading a medical image. Answer the question using "
    "ONLY what is visibly present in this image.\n"
    "Rules:\n"
    "- Do not mention any finding, organ or device you cannot actually see.\n"
    "- Do not guess, and do not infer from what is usually present in such images.\n"
    '- If the thing asked about is not clearly visible, or you are unsure, reply '
    'exactly: "{abstain}"\n\nQuestion: {question}'
)

# Copied verbatim from the adapter's train_info.json "prompt_template". The {q}
# placeholder is substituted literally rather than reformatted, so this is
# provably the string the adapter was fine-tuned on. Use prompt_style="train"
# to remove the train/inference prompt mismatch as a confound.
_TRAIN_PROMPT_TEMPLATE = "Answer the clinical question about this image briefly. {q}"

PROMPT_STYLES = ("plain", "strict", "train")


def default_prompt_style(mode: str) -> str:
    """Which prompt a mode uses when prompt_style is not given.

    baseline  -> "plain" (and no adapter).
    mitigated -> config.json's default_mitigated_prompt_style, "train" by
                 default: the exact template the adapter was fine-tuned on.
                 "strict" remains selectable per call but is deliberately not
                 the default - see the note on _MITIGATED_PROMPT.
    """
    if mode == "baseline":
        return "plain"
    style = load_config()["default_mitigated_prompt_style"]
    return style if style in PROMPT_STYLES else "train"


def build_prompt(question: str, mode: str, abstain_answer: str,
                 prompt_style: Optional[str] = None) -> str:
    """Render the user-turn text for one question."""
    style = prompt_style or default_prompt_style(mode)
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}, got {style!r}")
    if style == "train":
        return _TRAIN_PROMPT_TEMPLATE.replace("{q}", question)
    if style == "strict":
        return _MITIGATED_PROMPT.format(question=question, abstain=abstain_answer)
    return _BASELINE_PROMPT.format(question=question)


def _build_messages(image: Image.Image, question: str, mode: str, abstain_answer: str,
                    prompt_style: Optional[str] = None):
    text = build_prompt(question, mode, abstain_answer, prompt_style)
    return [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
    }]


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

def is_closed_question(question: str) -> bool:
    """Heuristic yes/no detector. predict() is not given the question_type."""
    return bool(_CLOSED_START.match(question)) or "yes or no" in question.lower()


def _first_token_ids(tokenizer, words: list[str]) -> list[int]:
    """First token id of each spelling, deduplicated."""
    ids = []
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return sorted(set(ids))


def _mean_token_probability(scores, sequences, prompt_len: int) -> Optional[float]:
    """Mean probability of the tokens the model actually generated."""
    import torch

    probs = []
    for step, step_logits in enumerate(scores):
        index = prompt_len + step
        if index >= sequences.shape[1]:
            break
        distribution = torch.softmax(step_logits[0].float(), dim=-1)
        probs.append(distribution[sequences[0, index]].item())
    if not probs:
        return None
    return sum(probs) / len(probs)


def _yes_no_confidence(first_step_logits, tokenizer,
                       min_mass: float = 0.10) -> Optional[float]:
    """Probability mass of yes vs no on the first generated token.

    Returns the winning side's share, so 0.5 means a coin flip and 1.0 means all
    of the yes/no mass sits on one side.

    Returns None when yes and no together hold less than `min_mass` of the
    distribution, because the ratio is scale-free and would otherwise report a
    confident-looking number off almost no probability. Measured on this model:
    a real "Yes" carries ~0.97 of the first-token mass, while a hedged prose
    answer carries ~0.03 and a refusal ~0.0005 - and the unguarded ratio turned
    those last two into 0.89 and 0.97. In mitigated mode that is the dangerous
    direction, since inflated confidence suppresses abstention. The caller falls
    back to the mean generated-token probability instead.
    """
    import torch

    distribution = torch.softmax(first_step_logits[0].float(), dim=-1)
    p_yes = distribution[_first_token_ids(tokenizer, _YES_WORDS)].sum().item()
    p_no = distribution[_first_token_ids(tokenizer, _NO_WORDS)].sum().item()
    total = p_yes + p_no
    if total < max(min_mass, 1e-6):
        return None
    return max(p_yes, p_no) / total


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


# --------------------------------------------------------------------------
# Stub fallback (no CUDA)
# --------------------------------------------------------------------------

def _stub_prediction(mode: str, abstain_answer: str) -> Prediction:
    if mode == "mitigated":
        return Prediction(abstain_answer, 0.5, True, f"[stub] {abstain_answer}", mode)
    return Prediction("yes", 0.5, False, "[stub] yes", mode)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def predict(
    image: Union[str, os.PathLike, Image.Image],
    question: str,
    mode: Mode = "baseline",
    adapter_path: Optional[str] = None,
    prompt_style: Optional[str] = None,
) -> Prediction:
    """Answer one clinical question about one image.

    mode="baseline"  - plain prompt, no adapter, never abstains.
    mode="mitigated" - abstains when the confidence falls below config.json's
                       abstention_threshold (0.8 as evaluated). Uses the
                       training template by default.
    adapter_path     - optional LoRA adapter directory, loaded with peft. The
                       evaluated system passes
                       model/checkpoints/medgemma_vqarad_lora.
    prompt_style     - override the prompt: "plain", "strict" or "train".
                       Default None keeps each mode's own prompt: baseline ->
                       "plain", mitigated -> "train" (the exact template the
                       adapter was fine-tuned on). "strict" is selectable but
                       is not the default; it abstains on ~79% of val
                       questions.
    """
    if mode not in ("baseline", "mitigated"):
        raise ValueError("mode must be 'baseline' or 'mitigated'")
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    image = image.convert("RGB")

    config = load_config()
    abstain_answer = config["abstain_answer"]

    if not cuda_available():
        return _stub_prediction(mode, abstain_answer)

    import torch

    model, processor = get_model(adapter_path=adapter_path, model_id=config["model_id"])
    messages = _build_messages(image, question, mode, abstain_answer, prompt_style)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    max_new_tokens = min(int(config["max_new_tokens"]), _MAX_NEW_TOKENS_CAP)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # greedy
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = generated.sequences
    scores = generated.scores or ()
    raw_text = processor.decode(sequences[0][prompt_len:], skip_special_tokens=True)
    answer = _clean(raw_text)

    # Record whether the numerics held up. A broken dtype choice shows up here
    # as non-finite logits and/or empty text, which no Prediction field can
    # express - callers check LAST_GENERATION and abort.
    non_finite_steps = sum(
        0 if bool(torch.isfinite(step_logits).all()) else 1 for step_logits in scores
    )
    LAST_GENERATION.update({
        "scores_finite": non_finite_steps == 0 and bool(scores),
        "non_finite_steps": non_finite_steps,
        "new_tokens": int(sequences.shape[1] - prompt_len),
        "text_empty": not raw_text.strip(),
        "dtype": str(compute_dtype(torch)),
        "device_capability": "%d.%d" % torch.cuda.get_device_capability(0),
    })

    confidence = None
    if scores:
        if is_closed_question(question):
            confidence = _yes_no_confidence(
                scores[0], processor.tokenizer,
                min_mass=float(config["min_yes_no_mass"]),
            )
        if confidence is None:
            confidence = _mean_token_probability(scores, sequences, prompt_len)
    # NaN must be caught before clamping: min()/max() compare False against NaN
    # and would quietly return 1.0, turning broken logits into max confidence.
    if confidence is None or not math.isfinite(float(confidence)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    abstained = False
    if mode == "mitigated":
        said_so = abstain_answer.lower() in answer.lower()
        if said_so or confidence < float(config["abstention_threshold"]):
            abstained = True
            answer = abstain_answer

    return Prediction(
        answer=answer,
        confidence=confidence,
        abstained=abstained,
        raw_text=raw_text,
        mode=mode,
        is_stub=False,
    )
