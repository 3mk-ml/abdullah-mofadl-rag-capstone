from __future__ import annotations

import re

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


def normalize_text(text: str) -> str:
    """Conservative normalization suitable for mixed Arabic/English technical text."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_bm25(text: str) -> str:
    text = normalize_text(text).lower()
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_for_bm25(text)
    return re.findall(r"[\w\-./%]+", text, flags=re.UNICODE)
