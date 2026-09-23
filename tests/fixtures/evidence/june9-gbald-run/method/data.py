"""Data loading.

Default behavior: `load_data()` (no args) reads from `method/example_data/` —
the directory bundled inside this package. The first call downloads a subset
of MNIST into that directory and saves a `.pt` file for fast reloading; later
calls just reload the file.

To use your own data, you have two options:

  - **Drop your file alongside the notebook** and pass the path explicitly:
    `load_data("my_data/")` or `load_data("my_dataset.pt")`.
  - **Drop your file inside `method/example_data/`** to override the default;
    `load_data()` (no args) will then pick it up.

Either way, your file must match one of these conventions — `load_data` picks
them up automatically, in priority order:

- **`.pt`** — single file containing a dict
  `{"x_pool": tensor, "y_pool": tensor, "x_test": tensor, "y_test": tensor}`.
- **`.json`** — single file with the same keys, values as nested lists. They
  are converted to tensors at load time.
- **`.csv`** — *two* files in the directory: `pool.csv` (or `train.csv`) and
  `test.csv`. Header row plus data rows; the **last column is the integer
  label**, all other columns are float features.

Inputs (`x_pool`, `x_test`) must be 2D `(N, input_dim)`. The bundled model
architecture (see `method/model.py`) expects flattened features. For
image-shaped data, swap in a CNN architecture in `method/model.py` and
remove the flatten step on input.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Format dispatchers
# ---------------------------------------------------------------------------


def _find_with_ext(p: Path, ext: str) -> Path | None:
    if p.is_file() and p.suffix == ext:
        return p
    if p.is_dir():
        for entry in sorted(p.iterdir()):
            if entry.suffix == ext:
                return entry
    return None


def _find_csv_pair(p: Path) -> tuple[Path | None, Path | None]:
    if not p.is_dir():
        return None, None
    by_stem = {entry.stem.lower(): entry for entry in p.iterdir() if entry.suffix == ".csv"}
    pool = by_stem.get("pool") or by_stem.get("train")
    test = by_stem.get("test")
    return pool, test


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------


def _load_pt(path: Path):
    d = torch.load(path, weights_only=False)
    return d["x_pool"], d["y_pool"], d["x_test"], d["y_test"]


def _load_json(path: Path):
    with path.open() as f:
        d = json.load(f)
    return (
        torch.tensor(d["x_pool"], dtype=torch.float),
        torch.tensor(d["y_pool"], dtype=torch.long),
        torch.tensor(d["x_test"], dtype=torch.float),
        torch.tensor(d["y_test"], dtype=torch.long),
    )


def _load_csv_pair(pool_csv: Path, test_csv: Path):
    def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open() as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = [row for row in reader if row]
        arr = np.array(rows, dtype=float)
        return arr[:, :-1], arr[:, -1]

    x_pool_np, y_pool_np = read(pool_csv)
    x_test_np, y_test_np = read(test_csv)
    return (
        torch.tensor(x_pool_np, dtype=torch.float),
        torch.tensor(y_pool_np, dtype=torch.long),
        torch.tensor(x_test_np, dtype=torch.float),
        torch.tensor(y_test_np, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# MNIST fallback
# ---------------------------------------------------------------------------


def _download_mnist_subset(
    save_dir: Path, pool_size: int, n_test: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Cache lives under `_cache/` so format dispatchers (which scan the data
    # dir for `.pt`/`.json`/`.csv`) skip it and user-supplied files at the top
    # level always take precedence. The cache stores its generation params so
    # a stale cache from an earlier (e.g. smoke-scale) call with different
    # sizes is regenerated rather than silently reused.
    cache_dir = save_dir / "_cache"
    cache_path = cache_dir / "mnist_subset.pt"
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, weights_only=False)
            meta = cached.get("_meta", {})
            if (
                meta.get("pool_size") == pool_size
                and meta.get("n_test") == n_test
                and meta.get("seed") == seed
            ):
                return cached["x_pool"], cached["y_pool"], cached["x_test"], cached["y_test"]
        except Exception:
            pass  # corrupt or schema-shifted cache — fall through and rebuild

    from torchvision import datasets, transforms

    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )

    full_train = datasets.MNIST(root=str(cache_dir / "_torchvision"), train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=str(cache_dir / "_torchvision"), train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(full_train), size=min(pool_size, len(full_train)), replace=False)
    test_idx = np.arange(min(n_test, len(full_test)))

    x_pool = torch.stack([full_train[int(i)][0] for i in pool_idx])
    y_pool = torch.tensor([full_train[int(i)][1] for i in pool_idx], dtype=torch.long)
    x_test = torch.stack([full_test[int(i)][0] for i in test_idx])
    y_test = torch.tensor([full_test[int(i)][1] for i in test_idx], dtype=torch.long)

    torch.save(
        {
            "x_pool": x_pool,
            "y_pool": y_pool,
            "x_test": x_test,
            "y_test": y_test,
            "_meta": {"pool_size": pool_size, "n_test": n_test, "seed": seed},
        },
        cache_path,
    )
    return x_pool, y_pool, x_test, y_test


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


DEFAULT_DATA_DIR = Path(__file__).parent / "example_data"


def load_data(
    path: str | os.PathLike | None = None,
    *,
    pool_size: int = 5000,
    n_test: int = 1000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_pool, y_pool, x_test, y_test).

    If `path` is None (the default), reads from `method/example_data/` — the
    directory bundled inside this package. Pass any path to read from a
    different location.

    Dispatches in priority order: .pt > .json > .csv (paired) > MNIST download.

    `pool_size`, `n_test`, and `seed` only apply when falling through to MNIST.
    User-supplied files are loaded with whatever sizes they were saved with;
    the MNIST cache (under `example_data/_cache/`) is regenerated whenever
    these params differ from what produced the cache.
    """
    p = DEFAULT_DATA_DIR if path is None else Path(path)

    # 1. .pt — fastest path
    pt = _find_with_ext(p, ".pt")
    if pt is not None:
        return _load_pt(pt)

    # 2. .json
    js = _find_with_ext(p, ".json")
    if js is not None:
        return _load_json(js)

    # 3. .csv pair (pool/train + test)
    pool_csv, test_csv = _find_csv_pair(p)
    if pool_csv is not None and test_csv is not None:
        return _load_csv_pair(pool_csv, test_csv)

    # 4. No data found — download MNIST as the default smoke dataset.
    return _download_mnist_subset(p if (p.is_dir() or not p.exists()) else p.parent, pool_size, n_test, seed)
