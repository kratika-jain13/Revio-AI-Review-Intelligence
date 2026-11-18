"""Complaint detection and professional reply generation for Revio AI.

Implemented: transparent rule-based issue detection and template-based replies.
Optional: an OpenAI-compatible LLM call when REVIO_USE_LLM=true and an API key
is available. If that call fails, the template fallback is always used.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

ISSUE_KEYWORDS: Dict[str, Dict[str, int]] = {
    "Delivery delay": {
        "delivery": 2, "late": 3, "delay": 3, "delayed": 3, "late tha": 3,
        "deeri": 3, "order": 1, "arrived": 1, "shipping": 2, "dispatch": 2,
        "minutes": 1, "hours": 1, "on time": 1,
    },
    "Poor customer support": {
        "customer support": 1, "support": 1, "respond": 1, "response": 1,
        "did not respond": 1, "no response": 2, "call": 1, "called": 1,
        "unhelpful": 2, "hung up": 2, "call nahi": 2, "reply": 1, "replied": 1,
    },
    "Product quality": {
        "product": 2, "quality": 2, "defective": 3, "damaged": 3, "broken": 3,
        "stopped working": 3, "poor quality": 3, "stale": 2, "kharab": 2,
    },
    "Billing / refund": {
        "bill": 2, "invoice": 2, "charged": 2, "overcharged": 3, "charge": 1,
        "refund": 3, "payment": 2, "charged twice": 3, "paise": 2,
        "wapas": 2, "billing": 2,
    },
    "Staff behavior": {
        "staff": 2, "manager": 2, "rude": 3, "behavior": 2, "behaviour": 2,
        "ignored": 2, "unhelpful": 1, "team": 1, "विनम्र": 1,
    },
    "App / website issue": {
        "app": 2, "website": 2, "login": 3, "crash": 3, "crashes": 3,
        "bug": 2, "stuck": 2, "search": 1, "page": 1, "website payment": 2,
    },
    "Cleanliness / hygiene": {
        "dirty": 3, "clean": 1, "hygiene": 3, "smell": 2, "packaging": 1,
        "गंदा": 3, "safai": 2,
    },
    "Appointment / scheduling": {
        "appointment": 3, "booking": 3, "slot": 3, "schedule": 2,
        "rescheduled": 3, "intimation": 2,
    },
    "Safety concern": {
        "safety": 4, "unsafe": 4, "hospital": 4, "injured": 4, "injury": 4,
        "allergic": 4, "poisoning": 4, "dangerous": 4, "damaged": 1,
    },
    "Pricing / value": {
        "price": 2, "expensive": 3, "cost": 2, "value": 2, "worth": 1,
        "reasonable": 1,
    },
}

# Used as a deterministic tie-breaker when two issues score the same.
ISSUE_ORDER = list(ISSUE_KEYWORDS.keys())

HIGH_PRIORITY_ISSUES = {
    "Safety concern", "Billing / refund", "Delivery delay",
    "Poor customer support", "Product quality",
}


def _contains(text: str, phrase: str) -> bool:
    lowered = text.lower()
    if re.fullmatch(r"[a-z0-9 ]+", phrase):
        pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
        return re.search(pattern, lowered) is not None
    return phrase.lower() in lowered


def detect_issues(text: str) -> Tuple[str, str, List[Tuple[str, int]]]:
    """Return (main_issue, secondary_issue, ranked_issues)."""
    if not text:
        return "General service", "None", []

    scores: Dict[str, int] = {}
    for issue, keywords in ISSUE_KEYWORDS.items():
        score = sum(weight for phrase, weight in keywords.items() if _contains(text, phrase))
        if score:
            scores[issue] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], ISSUE_ORDER.index(item[0])),
    )
    if not ranked:
        return "General service", "None", []
    main = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 else "None"
    return main, secondary, ranked


def determine_priority(sentiment: str, main_issue: str, rating: Optional[int] = None) -> str:
    """Map sentiment + issue severity to Low / Medium / High."""
    if sentiment == "Negative":
        if main_issue in HIGH_PRIORITY_ISSUES or (rating is not None and rating <= 2):
            return "High"
        return "Medium"
    if sentiment == "Neutral":
        return "Low"
    return "Low"


class ResponseGenerator:
    """Generate a professional, brand-safe reply with a template fallback."""

    def __init__(self, tone: str = "empathetic", use_llm: Optional[bool] = None) -> None:
        self.tone = tone
        flag = use_llm if use_llm is not None else os.getenv("REVIO_USE_LLM", "false").lower() == "true"
        self.use_llm = bool(flag)

    def generate(
        self,
        review_text: str,
        sentiment: str,
        main_issue: str,
        secondary_issue: str = "None",
        priority: str = "Low",
        rating: Optional[int] = None,
    ) -> Dict[str, str]:
        llm_reply = self._try_llm(review_text, sentiment, main_issue, secondary_issue, priority)
        if llm_reply:
            return {"reply": llm_reply, "source": "LLM API", "tone": self.tone}

        reply = self._template_reply(sentiment, main_issue, secondary_issue, priority)
        return {"reply": reply, "source": "template-based fallback", "tone": self.tone}

    def _template_reply(self, sentiment: str, main_issue: str, secondary_issue: str, priority: str) -> str:
        issue_text = main_issue.lower()
        if secondary_issue and secondary_issue != "None":
            issue_text += f" and {secondary_issue.lower()}"

        if sentiment == "Negative":
            if self.tone == "concise":
                reply = (
                    "We sincerely apologize for the inconvenience. We understand your concern "
                    f"regarding the {issue_text}. Our team will look into this issue."
                )
            elif self.tone == "professional":
                reply = (
                    "Thank you for bringing this to our attention. We apologize for the "
                    f"inconvenience caused by the {issue_text}. Our team is reviewing the matter "
                    "and will contact you with an update shortly."
                )
            else:
                reply = (
                    "We sincerely apologize for the inconvenience. We understand your concern "
                    f"regarding the {issue_text}. Our team will look into this issue and get back "
                    "to you shortly."
                )
            if main_issue == "Safety concern":
                reply += " This has been escalated to our safety lead for immediate review."
            elif priority == "High" and main_issue == "Billing / refund":
                reply += " We will verify the charge and share a refund update at the earliest."
            return reply

        if sentiment == "Positive":
            return (
                "Thank you for your kind words! We are glad you had a positive experience with "
                f"our {issue_text}. We truly appreciate you taking the time to share your feedback."
            )

        return (
            "Thank you for sharing your feedback. We have noted your comments about the "
            f"{issue_text} and will use them to improve our service."
        )

    def _try_llm(
        self,
        review_text: str,
        sentiment: str,
        main_issue: str,
        secondary_issue: str,
        priority: str,
    ) -> Optional[str]:
        if not self.use_llm or not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=os.getenv("REVIO_LLM_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                max_tokens=180,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short, empathetic public replies to customer reviews. "
                            "Never invent facts, refunds, or timelines. Keep it under 90 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Review: {review_text}\nSentiment: {sentiment}\n"
                            f"Main issue: {main_issue}\nSecondary issue: {secondary_issue}\n"
                            f"Priority: {priority}\nWrite the public reply only."
                        ),
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None
