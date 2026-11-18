"""Lightweight NLP preprocessing utilities (no model downloads required)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "so", "that",
    "the", "their", "them", "then", "there", "they", "this", "to", "was",
    "we", "were", "what", "when", "which", "who", "will", "with", "would",
    "you", "your", "very", "just", "about", "after", "all", "also", "had",
    "tha", "thi", "the", "hai", "hain", "nahi", "nahin", "ka", "ki", "ke",
    "ko", "mein", "se", "aur", "th", "ye", "yeh", "woh", "ek", "diya",
    "है", "हैं", "था", "थी", "थे", "में", "से", "और", "यह", "वह", "एक",
    "की", "के", "का", "को", "बहुत", "भी", "तो", "atha", "liya",
}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove URLs / social handles without killing sentiment cues."""
    if not text:
        return ""
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)
    cleaned = re.sub(r"@\w+|#\w+", " ", cleaned)
    cleaned = re.sub(r"[\u200b-\u200f\ufeff]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> List[str]:
    """Unicode-aware tokenization, including Devanagari words."""
    return re.findall(r"[\w\u0900-\u097F]+", clean_text(text).lower())


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def get_keywords(text: str, top_n: int = 10) -> List[tuple[str, int]]:
    """Return the most frequent meaningful tokens for 'frequently mentioned issues'."""
    tokens = remove_stopwords(tokenize(text))
    return Counter(tokens).most_common(top_n)


def detect_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    hindi_roman = {"tha", "thi", "hai", "nahi", "accha", "kharab", "paise", "bahut", "yaar"}
    hits = sum(1 for word in tokenize(text) if word in hindi_roman)
    return "Hinglish" if hits >= 2 else "English"
