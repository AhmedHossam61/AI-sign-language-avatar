"""Rule-based English text → ASL gloss converter.

Approach:
  1. Tokenize the sentence (spaCy if available, else regex fallback).
  2. Remove words that don't exist in ASL gloss (articles, auxiliaries, etc.).
  3. Lemmatize each token.
  4. Return uppercase gloss tokens.

Unknown tokens are returned as-is (uppercase) so the motion sequencer can
decide whether to fingerspell them or skip them.
"""

from __future__ import annotations

import re

# Words dropped entirely in ASL gloss (no sign equivalent at MVP stage).
_STOP_REMOVE: frozenset[str] = frozenset({
    # Articles
    "a", "an", "the",
    # Auxiliary / copula verbs
    "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did",
    # Conjunctions / filler
    "and", "or", "but", "so", "yet", "nor",
    "that", "which", "of", "to", "for", "in", "on", "at",
    "with", "by", "from", "up", "about", "into", "through",
    "during", "before", "after", "above", "below", "between",
    # Modal-ish that drop cleanly
    "can", "could", "would", "should", "shall", "may", "might", "must",
    # Other common filler
    "just", "very", "really", "quite", "also", "too", "only",
    "it", "its", "this", "these", "those",
    "there", "here",
    "'s", "n't",
})

# Simple suffix-stripping lemmatizer (no external library needed).
# Ordered from most-specific to least-specific.
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ies",   "y"),    # tries → try
    ("ied",   "y"),    # tried → try
    ("ying",  "y"),    # trying → try
    ("nning", "n"),    # running → run (double-consonant)
    ("tting", "t"),    # sitting → sit
    ("pping", "p"),    # stopping → stop
    ("dding", "d"),    # adding → add
    ("mming", "m"),    # swimming → swim
    ("ing",   ""),     # walking → walk
    ("ness",  ""),     # sadness → sad
    ("ful",   ""),     # helpful → help
    ("less",  ""),     # helpless → help (rough)
    ("tion",  ""),     # action (keep as-is after stripping)
    ("ed",    ""),     # walked → walk
    ("er",    ""),     # faster → fast (rough)
    ("est",   ""),     # fastest → fast (rough)
    ("ly",    ""),     # quickly → quick (rough)
    ("s",     ""),     # dogs → dog  (applied last, carefully)
]

# Manual overrides for common irregular forms.
_LEMMA_MAP: dict[str, str] = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "have": "have", "has": "have", "had": "have",
    "go": "go", "goes": "go", "went": "go", "gone": "go",
    "see": "see", "saw": "see", "seen": "see",
    "eat": "eat", "ate": "eat", "eaten": "eat",
    "drink": "drink", "drank": "drink", "drunk": "drink",
    "give": "give", "gave": "give", "given": "give",
    "know": "know", "knew": "know", "known": "know",
    "come": "come", "came": "come",
    "run": "run", "ran": "run",
    "sit": "sit", "sat": "sit",
    "write": "write", "wrote": "write", "written": "write",
    "take": "take", "took": "take", "taken": "take",
    "understand": "understand", "understood": "understand",
    "hear": "hear", "heard": "hear",
    "learn": "learn", "learned": "learn", "learnt": "learn",
    "speak": "speak", "spoke": "speak", "spoken": "speak",
    "stand": "stand", "stood": "stand",
    "sleep": "sleep", "slept": "sleep",
    "feel": "feel", "felt": "feel",
    "buy": "buy", "bought": "buy",
    "good": "good", "better": "good", "best": "good",
    "bad": "bad", "worse": "bad", "worst": "bad",
    "many": "more", "much": "more",
    "little": "less", "less": "less", "fewer": "less",
    "i": "me", "my": "me", "myself": "me",
    "he": "man", "him": "man", "his": "man",
    "she": "woman", "her": "woman", "hers": "woman",
    "they": "people", "them": "people", "their": "people",
    "we": "people", "us": "people", "our": "people",
    "you": "you", "your": "you", "yourself": "you",
}


def _simple_lemmatize(word: str) -> str:
    """Suffix-stripping lemmatizer — no external library required."""
    low = word.lower()
    if low in _LEMMA_MAP:
        return _LEMMA_MAP[low]

    for suffix, replacement in _SUFFIX_RULES:
        if low.endswith(suffix) and len(low) - len(suffix) >= 2:
            stem = low[: len(low) - len(suffix)] + replacement
            # Avoid over-stripping very short stems
            if len(stem) >= 2:
                return stem

    return low


def _tokenize_fallback(sentence: str) -> list[str]:
    """Simple regex tokenizer used when spaCy is not available."""
    # Split on whitespace and strip punctuation from token edges.
    raw = re.findall(r"[A-Za-z']+", sentence)
    return raw


def text_to_gloss(sentence: str) -> list[str]:
    """Convert an English sentence to a list of ASL gloss tokens.

    Returns a list of uppercase tokens. Tokens not found in the motion
    library are still returned — the sequencer decides what to do with them
    (fingerspell or skip).

    Args:
        sentence: Raw English input, e.g. "I want to eat pizza".

    Returns:
        List of uppercase gloss tokens, e.g. ["WANT", "EAT", "PIZZA"].
    """
    # Try spaCy for better lemmatization; fall back to our own tokenizer.
    try:
        import spacy  # type: ignore[import]
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
    except ImportError:
        nlp = None

    gloss: list[str] = []

    if nlp is not None:
        doc = nlp(sentence.lower())
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if token.text in _STOP_REMOVE or token.lemma_ in _STOP_REMOVE:
                continue
            lemma = token.lemma_.upper()
            if lemma:
                gloss.append(lemma)
    else:
        tokens = _tokenize_fallback(sentence)
        for tok in tokens:
            low = tok.lower()
            if low in _STOP_REMOVE:
                continue
            lemma = _simple_lemmatize(low).upper()
            if lemma:
                gloss.append(lemma)

    return gloss


if __name__ == "__main__":
    import sys
    sentence = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, how are you today?"
    print("Input:  ", sentence)
    print("Gloss:  ", text_to_gloss(sentence))
