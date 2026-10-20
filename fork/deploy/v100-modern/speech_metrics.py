"""Unicode word normalization and corpus-weighted ASR edit counts."""

import unicodedata


def words(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        c for c in text if not unicodedata.category(c).startswith(("P", "M"))
    ).split()


def errors(reference, hypothesis):
    """Levenshtein word errors; retain counts for corpus-weighted WER."""
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]
