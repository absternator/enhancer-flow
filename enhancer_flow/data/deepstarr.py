import gzip
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path
from Bio import SeqIO
import numpy as np
import pandas as pd

from .dataset import ConditionSpec, EnhancerDataset, seqs_to_dataset

BASE_URL = "https://data.starklab.org/almeida/DeepSTARR/Data"
SEQ_LEN = 249
SPLITS = ("Train", "Val", "Test")

# Column names in Sequences_activity_*.txt. The published files use these; if a
# future mirror renames them, override via the loader's ``activity_cols`` arg.
DEV_COL = "Dev_log2_enrichment"
HK_COL = "Hk_log2_enrichment"

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)

def download_deepstarr(data_dir: Path) -> Path:
    """Fetch all FASTA + activity files into ``data_dir/deepstarr``.

    Returns the directory. Idempotent — skips files already present. ~300 MB
    total; run once on the node before training.
    """
    for split in SPLITS:
        _download(f"{BASE_URL}/Sequences_{split}.fa", data_dir / f"Sequences_{split}.fa")
        _download(f"{BASE_URL}/Sequences_activity_{split}.txt", data_dir / f"Sequences_activity_{split}.txt")
    return data_dir

def _read_fasta(path: Path) -> list[str]:
    """Read a FASTA file and return a list of sequences."""
    return [str(record.seq) for record in SeqIO.parse(path, "fasta")]

def _read_activity(path: Path) -> np.ndarray:
    """Read dev/hk activity columns -> ``(N, 2)`` float array."""
    df = pd.read_table(path, sep="\t")
    if DEV_COL not in df.columns or HK_COL not in df.columns:
        raise ValueError(f"Missing expected columns in {path}: {df.columns}")
    return df[[DEV_COL, HK_COL]].to_numpy(dtype=np.float32)

def load_deepstarr(
    data_dir: str | Path,
    *,
    standardize: bool = True,
    download: bool = True,
) -> dict[str, EnhancerDataset]:
    root = Path(data_dir) / "deepstarr"
    if download:
        download_deepstarr(root)

    raw: dict[str, tuple[list[str], np.ndarray]] = {}
    for split in SPLITS:
        seqs = _read_fasta(root / f"Sequences_{split}.fa")
        activities = _read_activity(root / f"Sequences_activity_{split}.txt")
        if len(seqs) != len(activities):
            raise ValueError(f"Mismatch in {split}: {len(seqs)} sequences vs {len(activities)} activities")
        raw[split] = (seqs, activities)

    mean = np.zeros(2, dtype=np.float32)
    std = np.ones(2, dtype=np.float32)
    if standardize:
        train_act = raw["Train"][1]
        mean = train_act.mean(axis=0)
        std = train_act.std(axis=0) + 1e-8  # avoid division by zero
        (root / "cond_stats.json").write_text(
            json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)
        )

    spec = ConditionSpec("vector", dim=2)
    out: dict[str, EnhancerDataset] = {}
    for split, (seqs, activites) in raw.items():
        cond = (activites - mean) / std if standardize else activites
        out[split.lower()] = seqs_to_dataset(seqs, cond, spec, SEQ_LEN, name=f"deepstarr_{split.lower()}")
    return out

def load_cond_stats(data_dir: str | Path) -> dict[str, list[float]]:
    """Load the mean/std used to standardize DeepSTARR conditions."""
    return json.loads((Path(data_dir) / "deepstarr" / "cond_stats.json").read_text())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download + sanity-check DeepSTARR data")
    ap.add_argument("--data_dir", default="./data_cache")
    ap.add_argument("--no_download", action="store_true", help="skip download, just check existing files")
    args = ap.parse_args()
    ds = load_deepstarr(args.data_dir, download=not args.no_download)
    for name, d in ds.items():
        print(f"{name}: {len(d)} seqs, cond {d.conditions.shape}, "
              f"spec {asdict(d.cond_spec)}")
        print(f"  example: {d.decode_row(0)[:60]}...")