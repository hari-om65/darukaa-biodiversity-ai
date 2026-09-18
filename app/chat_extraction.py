"""Lightweight slot-filling: pulls the reasoning engine's required variables out
of free-form chat messages. Deliberately rule-based (not an LLM call) - it only
needs to recognize a small, known vocabulary of values, and keeping it
deterministic makes the clarification flow easy to test and free to run.
"""

import re
from typing import Any

REQUIRED_FIELDS = ["soil_organic_carbon", "rainfall", "crop", "region"]

CLARIFYING_QUESTIONS = {
    "soil_organic_carbon": "What is the soil organic carbon level, as a fraction (e.g. 0.3)?",
    "rainfall": "How would you describe recent rainfall - low, medium, or high?",
    "crop": "What crop or land-use pattern is in place (e.g. monoculture, polyculture, agroforestry, fallow)?",
    "region": "What kind of region is this (e.g. semi-arid, arid, temperate, tropical, humid)?",
}

_FLOAT_RE = r"(-?\d+(?:\.\d+)?)"

_SOC_KEYWORDS = ["soil organic carbon", "soc"]
_RAINFALL_KEYWORDS = ["rainfall", "precipitation"]
_RAINFALL_VALUES = ["low", "medium", "high"]
_CROP_KEYWORDS = ["crop", "land use", "land_use", "farming", "growing"]
_CROP_VALUES = ["monoculture", "polyculture", "agroforestry", "fallow"]
_REGION_KEYWORDS = ["region", "climate", "area", "zone"]
_REGION_VALUES = ["semi-arid", "arid", "humid", "temperate", "tropical", "semi-humid"]


def _extract_number_near(text: str, keywords: list[str]) -> float | None:
    for kw in keywords:
        match = re.search(rf"{re.escape(kw)}\D{{0,15}}?{_FLOAT_RE}", text)
        if match:
            return float(match.group(1))
    return None


def _extract_value(text: str, keywords: list[str], values: list[str]) -> str | None:
    # Prefer a value found near one of the keywords, but fall back to matching
    # the known-value vocabulary anywhere in the message.
    for kw in keywords:
        for val in values:
            if re.search(rf"{re.escape(kw)}.{{0,25}}?\b{re.escape(val)}\b", text):
                return val
    for val in values:
        if re.search(rf"\b{re.escape(val)}\b", text):
            return val
    return None


def extract_variables(message: str) -> dict[str, Any]:
    """Best-effort extraction of the required variables from a chat message."""
    text = message.lower()
    found: dict[str, Any] = {}

    soc = _extract_number_near(text, _SOC_KEYWORDS)
    if soc is not None:
        found["soil_organic_carbon"] = soc

    rainfall = _extract_value(text, _RAINFALL_KEYWORDS, _RAINFALL_VALUES)
    if rainfall:
        found["rainfall"] = rainfall

    crop = _extract_value(text, _CROP_KEYWORDS, _CROP_VALUES)
    if crop:
        found["crop"] = crop

    region = _extract_value(text, _REGION_KEYWORDS, _REGION_VALUES)
    if region:
        found["region"] = region

    return found


def missing_fields(variables: dict[str, Any]) -> list[str]:
    return [f for f in REQUIRED_FIELDS if variables.get(f) in (None, "")]


def build_clarifying_message(missing: list[str]) -> str:
    questions = [CLARIFYING_QUESTIONS[f] for f in missing]
    if len(questions) == 1:
        return questions[0]
    return "I need a bit more information:\n" + "\n".join(f"- {q}" for q in questions)
