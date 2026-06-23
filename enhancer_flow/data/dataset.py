
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from .encoding import VOCAB_SIZE, indices_to_seq, seq_to_indices

ConditionKind = Literal["vector", "class"]

@dataclass
class ConditionSpec:
    """Describes the conditioning signal so models can self-configure.

    For ``kind="vector"`` (DeepSTARR), ``dim`` is the number of scalar targets
    (2: dev, hk) and ``num_classes`` is unused. For ``kind="class"`` (fly-brain),
    ``num_classes`` is the number of categories and ``dim`` is the embedding-input
    width (1, a single id).
    """
    kind: ConditionKind
    dim: int
    num_classes: int = 0

    @property
    def is_vector(self) -> bool:
        return self.kind == "vector"

@dataclass
class EnhancerDataset:
    """In-memory enhancer dataset for one split.

    Attributes:
        indices: int8 array ``(N, L)`` of nucleotide ids (0..3), pad value 255.
        conditions: float32 array ``(N, cond_dim)`` of targets.
        cond_spec: how to interpret ``conditions``.
        seq_len: L (fixed across the dataset; shorter seqs are padded).
        name: split / dataset identifier for logging.
    """
    indices: np.ndarray
    conditions: np.ndarray
    cond_spec: ConditionSpec
    seq_len: int
    name: str = "unnamed"

    def __post_init__(self) -> None:
        n_seq = self.indices.shape[0]
        if self.conditions.shape[0] != n_seq:
            raise ValueError(
                f"{self.name}: {n_seq} sequences but "
                f"{self.conditions.shape[0]} conditions"
            )

    def __len__(self) -> int:
        return self.indices.shape[0]

    # ----------------- encode/decode ------------------
    def onehot_batch(self, rows: np.ndarray) -> np.ndarray:
        """Expand selected rows to simplex one-hots ``(B, L, 4)`` float32.

        Pad positions (id 255) become the uniform simplex point so they read as
        "unknown" rather than biasing toward any base.
        """
        idx = self.indices[rows].astype(np.int64) # (B, L)
        b, length = idx.shape
        out = np.zeros((b, length, VOCAB_SIZE), dtype=np.float32)
        valid = (idx >= 0) & (idx < VOCAB_SIZE)
        bb, ll = np.nonzero(valid)
        out[bb, ll, idx[valid]] = 1.0
        #uniform with pads
        out[~valid] = 1.0 / VOCAB_SIZE
        return out

    def decode_row(self, row: int) -> str:
        idx = self.indices[row]
        idx = idx[(idx >= 0) & (idx < VOCAB_SIZE)]
        return indices_to_seq(idx)

    # ----------------- iteration ------------------
    def iter_batchs(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True
    ) -> Iterator[dict[str, np.ndarray]]:
        """
        Yield dict batches with simplex sequences + conditions.

        Each batch: ``{"x1": (B, L, 4) f32, "cond": (B, cond_dim) f32}`` where
        ``x1`` is the *clean* target the flow path concentrates toward.
        """
        n = len(self)
        order = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        stop = n - (n % batch_size) if drop_last else n
        for start in range(0, stop, batch_size):
            rows = order[start : start + batch_size]
            yield {
                "x1": self.onehot_batch(rows),
                "cond": self.conditions[rows].astype(np.float32)
            }

def make_synthetic(
    n: int = 256,
    seq_len: int = 249,
    cond_spec: ConditionSpec | None = None,
    seed: int = 0
) -> EnhancerDataset:
    """Tiny random dataset for smoke tests / CI (no download needed).

    Sequences are random nucleotides; conditions are random standard normals
    (vector) so the whole pipeline can run end-to-end without real data. This is
    what the unit tests and the ``--smoke`` training run use.
    """
    rng = np.random.default_rng(seed)
    spec = cond_spec or ConditionSpec(kind="vector", dim=2)
    indices = rng.integers(0, VOCAB_SIZE, size=(n, seq_len)).astype(np.int8)
    if spec.is_vector:
        conditions = rng.standard_normal((n, spec.dim)).astype(np.float32)
    else:
        conditions = rng.integers(0, spec.num_classes, (n, 1)).astype(np.float32)
    return EnhancerDataset(indices, conditions, spec, seq_len, name="synthetic")

def seqs_to_dataset(
    seqs: list[str],
    conditions: np.ndarray,
    cond_spec: ConditionSpec,
    seq_len: int,
    name: str = "custom",
) -> EnhancerDataset:
    """Build a dataset from raw strings (used by the DeepSTARR loader)."""
    pad = 255
    idx_arr = np.full((len(seqs), seq_len), pad, dtype=np.int16)
    for i, s in enumerate(seqs):
        ids = seq_to_indices(s[:seq_len])
        ids[ids < 0] = 0  # treat unknown as A for storage; rare in this data
        idx_arr[i, : len(ids)] = ids
    # store as int16 because pad value 255 fits, but keep headroom; cast in batch
    return EnhancerDataset(
        idx_arr.astype(np.int16), conditions.astype(np.float32), cond_spec, seq_len, name
    )