"""Scorers map (model output, expected answer) -> score in [0, 1]."""
import re

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract_number(text: str) -> float | None:
    """Prefer the number after 'Answer:'; fall back to the last number in the text."""
    m = re.search(rf"answer\s*[:=]?\s*\$?\s*({_NUMBER})", text, re.IGNORECASE)
    matches = [m.group(1)] if m else re.findall(_NUMBER, text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def numeric_match(output: str, expected) -> float:
    pred = extract_number(output)
    return float(pred is not None and abs(pred - float(expected)) < 1e-6)


def exact_match(output: str, expected) -> float:
    return float(output.strip().lower() == str(expected).strip().lower())


SCORERS = {
    "numeric_match": numeric_match,
    "exact_match": exact_match,
}
