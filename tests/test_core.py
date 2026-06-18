import jax
import jax.numpy as jnp
import numpy as np

from enhancer_flow.data import (
    ALPHABET, VOCAB_SIZE, onehot_to_seq, seq_to_indices, seq_to_onehot,
)

def test_encode_decode_roundtrip():
    seq = "ACGT" * 10 + "ACGTACG"
    oh = seq_to_onehot(seq)
    assert oh.shape == (len(seq), VOCAB_SIZE)
    assert np.allclose(oh.sum(-1), 1.0)
    assert onehot_to_seq(oh) == seq


def test_unknown_base_is_uniform():
    oh = seq_to_onehot("ANCG")
    assert np.allclose(oh[1], 1.0 / VOCAB_SIZE)  # N -> uniform


def test_indices_match_alphabet():
    idx = seq_to_indices("ACGT")
    assert list(idx) == [ALPHABET.index(c) for c in "ACGT"]