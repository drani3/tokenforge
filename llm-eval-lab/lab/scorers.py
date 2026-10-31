"""Scorers map (model output, expected answer) -> a dict of fields merged into the result row.

Every scorer returns at least `score` in [0, 1]. Numeric scoring follows lm-evaluation-harness
and reports two views:
    strict_score  the answer must appear in the requested 'Answer: <number>' format
    score         flexible: falls back to the last number in the response if the format is missing
"""
import re

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
# 'Answer: 14', '**Answer:** $14', 'answer = 14'. Requires ':' or '=' so prose like
# "here's the answer: we have 12 oranges" doesn't match (the number must follow directly).
_ANSWER = re.compile(rf"answer\s*\**\s*[:=]\s*\**\s*\$?\s*({_NUMBER})", re.IGNORECASE)


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str) -> tuple[float | None, str]:
    """Return (number, method): method is 'format' (last 'Answer: n'), 'fallback' (last number), or 'none'."""
    if matches := _ANSWER.findall(text):
        return _to_float(matches[-1]), "format"
    if numbers := re.findall(_NUMBER, text):
        return _to_float(numbers[-1]), "fallback"
    return None, "none"


def _equal(pred: float | None, expected) -> bool:
    return pred is not None and abs(pred - float(expected)) < 1e-6


def numeric_match(output: str, expected) -> dict:
    pred, method = extract_answer(output)
    correct = _equal(pred, expected)
    return {
        "score": float(correct),
        "strict_score": float(correct and method == "format"),
        "extracted": pred,
        "extraction": method,
    }


def exact_match(output: str, expected) -> dict:
    return {"score": float(output.strip().lower() == str(expected).strip().lower())}


SCORERS = {
    "numeric_match": numeric_match,
    "exact_match": exact_match,
}
