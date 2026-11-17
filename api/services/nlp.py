"""Small gloss-to-English rendering helpers for isolated-sign predictions."""

from __future__ import annotations

import re


_ENGLISH_GLOSS_OVERRIDES = {
    "big_large": "Big or large",
    "good_morning": "Good morning",
    "i": "I",
    "small_little": "Small or little",
    "store_or_shop": "Store or shop",
    "thank_you": "Thank you",
    "train_ticket": "Train ticket",
    "tshirt": "T-shirt",
    "you_plural": "You",
}

_GLOSS_ALIASES = {
    "big_large": ("big", "large", "big or large"),
    "cell_phone": ("cell phone", "mobile phone", "phone"),
    "good_morning": ("good morning",),
    "small_little": ("small", "little", "small or little"),
    "store_or_shop": ("store", "shop", "store or shop"),
    "thank_you": ("thank you", "thanks"),
    "train_ticket": ("train ticket",),
    "tshirt": ("t shirt", "tee shirt"),
    "you_plural": ("you",),
}

_SUPPORTED_GLOSSES = {
    "bank", "big_large", "bird", "black", "boy", "brother", "car", "cell_phone",
    "court", "cow", "death", "dog", "dry", "election", "fall", "fan", "father",
    "girl", "good", "good_morning", "happy", "hat", "hello", "hot", "house",
    "i", "it", "long", "loud", "monday", "new", "no", "paint", "pen", "priest",
    "quiet", "red", "shoes", "short", "small_little", "store_or_shop", "summer",
    "teacher", "thank_you", "time", "train_ticket", "tshirt", "white", "window",
    "year", "yes", "you_plural",
}


def gloss_to_english(gloss: str | None) -> str | None:
    """Render one model label as readable English; this is not sentence translation."""

    if gloss is None:
        return None

    normalized_gloss = gloss.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized_gloss:
        return None

    override = _ENGLISH_GLOSS_OVERRIDES.get(normalized_gloss)
    if override is not None:
        return override
    return normalized_gloss.replace("_", " ").capitalize()


_TEXT_TO_GLOSS = {
    phrase: gloss
    for gloss in _SUPPORTED_GLOSSES
    for phrase in (gloss.replace("_", " "),)
} | {
    alias: gloss for gloss, aliases in _GLOSS_ALIASES.items() for alias in aliases
}


def text_to_glosses(text: str) -> tuple[list[str], list[str]]:
    """Match known words/phrases to labels; report unmatched tokens without guessing."""

    normalized_text = re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()
    tokens = normalized_text.split()
    glosses: list[str] = []
    unmatched: list[str] = []
    position = 0

    while position < len(tokens):
        matched_gloss: str | None = None
        matched_length = 0
        for phrase, gloss in sorted(
            _TEXT_TO_GLOSS.items(), key=lambda item: len(item[0].split()), reverse=True
        ):
            phrase_tokens = phrase.split()
            if tokens[position : position + len(phrase_tokens)] == phrase_tokens:
                matched_gloss = gloss
                matched_length = len(phrase_tokens)
                break

        if matched_gloss is not None:
            glosses.append(matched_gloss)
            position += matched_length
            continue

        token = tokens[position]
        if token in {"a", "an", "the", "please"}:
            position += 1
            continue
        unmatched.append(token)
        position += 1

    return glosses, unmatched