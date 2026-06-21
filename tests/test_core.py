import jax
import jax.numpy as jnp
import numpy as np

from enhancer_flow.data import (
    ALPHABET, VOCAB_SIZE, onehot_to_seq, seq_to_indices, seq_to_onehot, make_synthetic, ConditionSpec
)
from enhancer_flow.data.dataset import seqs_to_dataset

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

def test_make_synthetic():
    dataset = make_synthetic(100, 10)

    iterator = dataset.iter_batchs(batch_size=5, shuffle=False, drop_last=False)
    batch = next(iterator)

    assert len(dataset) == 100
    assert batch["x1"].shape == (5, 10, VOCAB_SIZE)
    assert batch["cond"].shape == (5, dataset.cond_spec.dim)
    assert np.allclose(batch["x1"].sum(axis=-1), 1.0)  # one-hot or uniform


def test_seqs_to_dataset():
    seqs = ["ACGT", "TGCA", "NNNN"]
    conditions = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    dataset = seqs_to_dataset(seqs, conditions, ConditionSpec("vector", dim=1), seq_len=10)

    assert len(dataset) == 3
    assert dataset.seq_len == 10
    assert dataset.conditions.shape == (3, 1)
    assert np.allclose(dataset.onehot_batch(np.arange(3)).sum(axis=-1), 1.0)