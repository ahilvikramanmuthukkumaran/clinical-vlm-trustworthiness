"""Public interface of the model module.

This is a STUB. It lets the evaluation / Streamlit side be built now.
The real implementation replaces the body of `predict()` later; the signature
and the Prediction fields stay the same (see model/docs/INTERFACE.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Union

from PIL import Image

Mode = Literal["baseline", "mitigated"]


@dataclass
class Prediction:
    answer: str            # final answer text shown to the user
    confidence: float      # 0.0 - 1.0 (stub: always 0.5)
    abstained: bool        # True if the model declined to answer
    raw_text: str          # unprocessed model generation
    mode: str              # "baseline" or "mitigated"
    is_stub: bool = True   # False once the real model is wired in

    def to_dict(self) -> dict:
        return asdict(self)


def predict(image: Union[str, Image.Image], question: str, mode: Mode = "baseline") -> Prediction:
    if mode not in ("baseline", "mitigated"):
        raise ValueError("mode must be 'baseline' or 'mitigated'")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if mode == "mitigated":
        return Prediction("cannot determine from this image", 0.5, True,
                          "[stub] cannot determine from this image", mode)
    return Prediction("yes", 0.5, False, "[stub] yes", mode)
