from __future__ import annotations
import numpy as np

# Canonical alphabet. DO NOT reorder — the oracle and motif tools assume A,C,G,T.
ALPHABET = "ACGT"

VOCAB_SIZE = len(ALPHABET)
_CHAR_TO_IDX = {c: i for i,c in enumerate(ALPHABET)}
_UNIFORM = np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE, dtype=np.float32)

def seq_to_indices(seq: str) -> np.ndarray:
    """``str`` of length L -> int array ``(L,)`` in 0..3, with -1 for unknowns."""
    return np.array([ _CHAR_TO_IDX.get(c.upper(), -1) for c in seq], dtype=np.int32)

def seq_to_onehot(seq: str) -> np.ndarray:
    """``str`` of length L -> simplex array ``(L, 4)`` float32.

    Known bases become one-hot vertices; unknown bases (N etc.) become the
    uniform point so the encoding is always a valid simplex coordinate.
    """
    idx = seq_to_indices(seq)
    out = np.tile(_UNIFORM, (len(seq), 1)).copy()
    known = idx >= 0
    out[known] = 0.0
    out[known, idx[known]] = 1.0
    return out

def onehot_to_seq(onehot: np.ndarray) -> str:
    """Simplex/logit array ``(L, 4)`` -> ``str`` via per-position argmax.

    Works on probabilities, one-hots, or raw logits — only the argmax matters.
    Note: does not handle unkown bases (N) gracefully, since the argmax will always pick one of A/C/G/T.
    """
    if onehot.ndim != 2 or onehot.shape[1] != VOCAB_SIZE:
        raise ValueError(f"Expected shape (L, {VOCAB_SIZE}), got {onehot.shape}")
    idx = np.argmax(onehot, axis=-1)
    return "".join(ALPHABET[i] for i in idx)

def indices_to_seq(idx: np.ndarray) -> str:
    """Int array ``(L,)`` in 0..3 -> ``str``.
    Note: does not handle unknown bases (N) gracefully, it always picks one of A/C/G/T.
    """
    return "".join(ALPHABET[int(i)] for i in idx)

def gc_content(seq: str) -> float:
    """Fraction of G/C bases — a cheap composition sanity check used in eval.
    """
    if not seq:
        return 0.0
    gc_count = sum(1 for c in seq if c.upper() in "GC")
    return gc_count / len(seq) if len(seq) > 0 else 0.0
