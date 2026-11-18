"""
Sentiment analysis for Revio AI.

Default behaviour:
  * Try a pretrained Hugging Face transformer (DistilBERT SST-2).
  * If the library/model cannot be loaded (offline demo, no internet, missing
    weights), fall back to a transparent lexicon model that also understands
    common Hindi / Hinglish review language.

Set environment variables to control the backend:
  REVIEW_SENTIMENT_BACKEND = auto | hf | lexicon
  REVIEW_SENTIMENT_MODEL   = any Hugging Face sentiment model id
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

POSITIVE_WORDS = {
    "excellent": 3.0, "amazing": 3.0, "wonderful": 2.8, "great": 2.5,
    "best": 2.5, "love": 2.5, "loved": 2.5, "delicious": 2.5, "fresh": 1.5,
    "friendly": 2.0, "helpful": 2.0, "polite": 1.8, "professional": 1.8,
    "clean": 1.5, "quick": 1.5, "fast": 1.5, "smooth": 1.5, "satisfied": 2.0,
    "happy": 2.0, "recommend": 2.0, "good": 1.5, "perfect": 2.5,
    "responsive": 1.8, "resolved": 1.8, "neat": 1.2, "reasonable": 1.2,
    "badhiya": 2.0, "badiya": 2.0, "bahut accha": 2.5, "accha": 1.5,
    "mast": 2.0, "pasand": 1.8, "khush": 1.8,
    "अच्छा": 2.0, "बहुत अच्छा": 2.5, "स्वादिष्ट": 2.5,
    "विनम्र": 2.0, "बढ़िया": 2.0, "धन्यवाद": 1.0, "संतुष्ट": 2.0,
}

NEGATIVE_WORDS = {
    "worst": 3.5, "terrible": 3.0, "awful": 3.0, "horrible": 3.0,
    "pathetic": 3.0, "disappointing": 2.5, "disappointed": 2.5, "rude": 2.5,
    "bad": 2.0, "poor": 2.0, "dirty": 2.0, "damaged": 2.5, "defective": 2.5,
    "late": 2.0, "delay": 2.0, "delayed": 2.0, "slow": 1.8, "wrong": 1.8,
    "missing": 1.8, "overcharged": 2.5, "charged twice": 3.0,
    "refund": 1.5, "complaint": 1.5, "ignored": 2.0, "unhelpful": 2.0,
    "stuck": 1.8, "crash": 2.0, "crashes": 2.0, "bug": 1.5,
    "fraud": 3.0, "scam": 3.0, "cheated": 3.0, "hospital": 3.0,
    "unsafe": 3.0, "safety": 2.0, "injury": 3.0, "injured": 3.0,
    "kharab": 2.5, "bura": 2.0, "bakwas": 2.5, "dhoka": 3.0,
    "खराब": 2.5, "बुरा": 2.0, "देरी": 2.0, "गंदा": 2.0, "निराश": 2.0,
}

NEGATORS = {"not", "no", "never", "nahi", "nahin", "nahiin", "नहीं"}


def _contains(text: str, phrase: str) -> bool:
    lowered = text.lower()
    if re.fullmatch(r"[a-z0-9 ]+", phrase):
        pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
        return re.search(pattern, lowered) is not None
    return phrase.lower() in lowered


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097F]+", text.lower())


def _lexicon_scores(text: str) -> tuple[float, float, list[str]]:
    tokens = _tokens(text)
    positive = 0.0
    negative = 0.0
    evidence: list[str] = []

    for word, weight in POSITIVE_WORDS.items():
        if _contains(text, word):
            word_tokens = _tokens(word)
            negated = False
            if word_tokens and word_tokens[0] in tokens:
                index = tokens.index(word_tokens[0])
                negated = any(token in NEGATORS for token in tokens[max(0, index - 2):index + 1])
            if negated:
                negative += weight * 0.8
                evidence.append(f"not {word}")
            else:
                positive += weight
                evidence.append(word)

    for word, weight in NEGATIVE_WORDS.items():
        if _contains(text, word):
            negative += weight
            evidence.append(word)

    denominator = positive + negative + 2.5
    polarity = (positive - negative) / denominator
    return polarity, min(0.97, 0.5 + abs(polarity) * 0.45 + min(len(evidence), 6) * 0.02), evidence


def _map_hf_label(label: str, score: float) -> tuple[str, float]:
    normalized = label.upper().strip()
    if "POS" in normalized or normalized in {"LABEL_2", "LABEL_4", "LABEL_5"} or "5 STAR" in normalized or "4 STAR" in normalized:
        return "Positive", score
    if "NEG" in normalized or normalized in {"LABEL_0", "LABEL_1"} or "1 STAR" in normalized or "2 STAR" in normalized:
        return "Negative", score
    if "NEU" in normalized or normalized in {"LABEL_3", "3 STAR"}:
        return "Neutral", score
    return ("Positive" if score >= 0.5 else "Negative"), score


class SentimentAnalyzer:
    """Hybrid sentiment analyzer with transformer-first, lexicon-fallback design."""

    def __init__(self, model_name: Optional[str] = None, backend: Optional[str] = None) -> None:
        self.model_name = model_name or os.getenv("REVIEW_SENTIMENT_MODEL", DEFAULT_MODEL)
        self.backend = (backend or os.getenv("REVIEW_SENTIMENT_BACKEND", "auto")).lower()
        self._pipe: Any = None
        self.active_model = "lexicon-fallback"
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "lexicon":
            self.active_model = "lexicon-fallback"
            return
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
            )
            self.active_model = self.model_name
        except Exception as exc:  # offline / missing package / bad model id
            if self.backend == "hf":
                raise RuntimeError(f"Could not load Hugging Face model '{self.model_name}': {exc}") from exc
            self._pipe = None
            self.active_model = "lexicon-fallback (Hugging Face unavailable)"

    def analyze(self, text: str, rating: Optional[int] = None) -> Dict[str, Any]:
        """Return sentiment, confidence, model name and negative-feedback flag."""
        if not text or not text.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "model": self.active_model,
                "negative_feedback": False,
                "scores": {"Positive": 0.0, "Negative": 0.0, "Neutral": 1.0},
                "evidence": [],
            }

        polarity, lex_conf, evidence = _lexicon_scores(text)

        if self._pipe is not None:
            try:
                raw = self._pipe(text[:512])[0]
                base_label, base_score = _map_hf_label(raw["label"], float(raw["score"]))
                if base_label == "Positive":
                    scores = {"Positive": base_score, "Negative": 1 - base_score, "Neutral": 0.0}
                    signal = base_score
                elif base_label == "Negative":
                    scores = {"Positive": 1 - base_score, "Negative": base_score, "Neutral": 0.0}
                    signal = -base_score
                else:
                    scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": base_score}
                    signal = 0.0
            except Exception:
                self._pipe = None
                self.active_model = "lexicon-fallback (model call failed)"
                base_label, base_score, scores, signal = self._fallback_scores(polarity, lex_conf)
        else:
            base_label, base_score, scores, signal = self._fallback_scores(polarity, lex_conf)

        # Blend a small amount of star-rating signal so a 1-star "good product,
        # terrible support" review is not misread as purely positive.
        if rating is not None:
            rating_signal = (int(rating) - 3) / 2
            combined = 0.78 * signal + 0.22 * rating_signal
        else:
            combined = signal

        if combined > 0.18:
            sentiment = "Positive"
        elif combined < -0.18:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        confidence = round(min(0.99, max(0.35, abs(combined) * 0.7 + base_score * 0.3)), 3)
        scores[sentiment] = confidence
        total = sum(scores.values()) or 1
        scores = {k: round(v / total, 3) for k, v in scores.items()}

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "model": self.active_model,
            "negative_feedback": sentiment == "Negative",
            "scores": scores,
            "evidence": evidence[:8],
        }

    @staticmethod
    def _fallback_scores(polarity: float, confidence: float) -> tuple[str, float, dict, float]:
        if polarity > 0.12:
            label = "Positive"
            scores = {"Positive": confidence, "Negative": 1 - confidence, "Neutral": 0.0}
            signal = confidence
        elif polarity < -0.12:
            label = "Negative"
            scores = {"Positive": 1 - confidence, "Negative": confidence, "Neutral": 0.0}
            signal = -confidence
        else:
            label = "Neutral"
            scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": confidence}
            signal = 0.0
        return label, confidence, scores, signal
