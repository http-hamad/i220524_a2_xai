"""
Production-style moderation guardrail pipeline (Layer 1 regex, Layer 2 calibrated model, Layer 3 review).
This module is imported by the Part 5 notebook and can be run as a library.
"""

from __future__ import annotations

import os

os.environ["USE_TF"] = "0"

import re
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BLOCKLIST = {
    "direct_threat": [
        re.compile(
            r"\b(?:i|we)\s+(?:will|'ll|am\s+going\s+to|am\s+gonna|gonna|going\s+to)\s+(?:kill|murder|shoot|stab|hurt|beat|destroy)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:you|u)\s+(?:are|'re)\s+going\s+to\s+(?:die|get\s+hurt|suffer)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:someone|everybody|we)\s+should\s+(?:kill|hurt|shoot|stab)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+(?:will|'ll)\s+find\s+(?:where\s+you\s+live|your\s+(?:home|address))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|i'm|i\s+am)\s+going\s+to\s+(?:hurt|kill|end)\s+(?:you|u)\b",
            re.IGNORECASE,
        ),
    ],
    "self_harm_directed": [
        re.compile(
            r"\b(?:you|u)\s+should\s+(?:kill|hurt)\s+(?:yourself|yourselves)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bgo\s+(?:kill|hurt)\s+yourself\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:nobody|no\s+one)\s+would\s+miss\s+you\s+if\s+you\s+(?:died|were\s+gone)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdo\s+(?:everyone|everybody)\s+a\s+favou?r\s+and\s+(?:disappear|die)\b",
            re.IGNORECASE,
        ),
    ],
    "doxxing_stalking": [
        re.compile(
            r"\b(?:i|we)\s+(?:know|knew)\s+where\s+you\s+live\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+(?:will|'ll|am\s+going\s+to)\s+post\s+your\s+(?:address|phone|number)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+found\s+your\s+(?:real\s+name|address|phone)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:everyone|everybody)\s+will\s+know\s+who\s+you\s+really\s+are\b",
            re.IGNORECASE,
        ),
    ],
    "dehumanization": [
        re.compile(
            r"\b(?:you\s+people|those\s+people|they)\s+are\s+(?:not\s+)?(?:human|people|person)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:you\s+people|those\s+people|they)\s+are\s+animals\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:you\s+people|those\s+people|they)\s+should\s+be\s+exterminated\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:you\s+people|those\s+people|they)\s+are\s+a\s+disease\b",
            re.IGNORECASE,
        ),
    ],
    "coordinated_harassment": [
        re.compile(
            r"\b(?:everyone|everybody)\s+report\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\blet(?:'s|s| us)\s+all\s+go\s+after\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\braid\s+(?=their\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bmass\s+report\s+this\s+account\b",
            re.IGNORECASE,
        ),
    ],
}


def input_filter(text: str) -> Optional[Dict[str, Any]]:
    """Return a block decision dict if any compiled pattern matches; otherwise None."""
    if text is None:
        return None
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


class HFProbabilityEstimator:
    """Sklearn-style adapter so CalibratedClassifierCV can calibrate transformer probabilities."""

    def __init__(self, model: Any, tokenizer: Any, device: str, max_length: int = 128) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def fit(self, X: Any, y: Any):
        """No-op fit for pre-trained base estimators used with cv='prefit'."""
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return shape (n, 2) probabilities for non-toxic / toxic ordering as sklearn convention."""
        if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
            texts = list(X)
        else:
            texts = [str(x) for x in X]
        self.model.eval()
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs


class ModerationPipeline:
    """
    Three-layer moderation pipeline:
    Layer 1: fast regex blocklist
    Layer 2: calibrated transformer probabilities with allow/block/review bands
    Layer 3: human review queue for uncertain band
    """

    def __init__(
        self,
        model_dir: str,
        calibrator_path: str,
        device: Optional[str] = None,
        max_length: int = 128,
        block_hi: float = 0.6,
        allow_lo: float = 0.4,
    ) -> None:
        self.model_dir = model_dir
        self.calibrator_path = calibrator_path
        self.max_length = max_length
        self.block_hi = block_hi
        self.allow_lo = allow_lo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.isotonic = joblib.load(calibrator_path)

    def _raw_toxic_prob(self, text: str) -> float:
        """Return uncalibrated softmax probability for the toxic class (index 1)."""
        est = HFProbabilityEstimator(self.model, self.tokenizer, self.device, self.max_length)
        proba = est.predict_proba([text])
        return float(proba[0, 1])

    def _calibrated_toxic_prob(self, text: str) -> float:
        """Map raw toxic probability through an isotonic calibrator fit on held-out scores."""
        raw = self._raw_toxic_prob(text)
        return float(np.clip(self.isotonic.predict(np.array([raw], dtype=float))[0], 0.0, 1.0))

    def predict(self, text: str) -> Dict[str, Any]:
        """Run layers in order and return a structured decision dictionary."""
        blocked = input_filter(text)
        if blocked is not None:
            return dict(blocked)
        p_toxic = self._calibrated_toxic_prob(text)
        if p_toxic >= self.block_hi:
            return {"decision": "block", "layer": "model", "confidence": p_toxic}
        if p_toxic <= self.allow_lo:
            return {"decision": "allow", "layer": "model", "confidence": p_toxic}
        return {"decision": "review", "layer": "model", "confidence": p_toxic}
