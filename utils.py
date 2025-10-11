"""
utils.py
Misc helpers: device selection, timing, filename parsing, and seeding.

This module provides small utilities used across experiments:

- select_device(pref): pick a CUDA device if available, otherwise CPU.
- seed_all(seed): seed python, numpy and torch (including CUDA if present).
- ensure_dir(path): create a directory and return a Path.
- parse_epoch_acc(filename): parse names like 'epoch03_acc0.842.pt'.
- Timer: simple context manager to measure elapsed seconds.

Doctests are provided for `parse_epoch_acc`.
"""

from __future__ import annotations

import re
import time
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


def select_device(pref: Optional[int] = None) -> torch.device:
    """Select a torch.device.

    Prefers CUDA device index `pref` if provided and available. If `pref` is
    None, picks `cuda:0` when any GPU is present. Falls back to CPU.

    Prints the chosen device once and returns it.
    """
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if pref is not None:
            if 0 <= pref < n:
                dev = torch.device(f"cuda:{pref}")
            else:
                # requested index out of range -> pick 0
                dev = torch.device("cuda:0")
        else:
            # pick the first available CUDA device
            dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    print(f"Using device: {dev}")
    return dev


def seed_all(seed: int = 1337) -> None:
    """Seed python, numpy and torch (including CUDA). Set deterministic flags.

    Notes:
        - Sets `torch.backends.cudnn.deterministic = True` and
          `torch.backends.cudnn.benchmark = False` to favor determinism. This
          may slow training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Favor deterministic behavior where reasonable.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Older/stripped torch builds may not expose these attributes.
        pass


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure `path` exists as a directory and return a `Path` object.

    Example:
        >>> p = ensure_dir('tmp/some_dir')
        >>> isinstance(p, Path)
        True
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_epoch_acc(filename: str) -> Tuple[int, Optional[float]]:
    """Parse filenames like 'epoch03_acc0.842.pt' and return (epoch, acc).

    The function is robust to separators ('_', '-', none) and extra text.

    Returns:
        (epoch, accuracy) where accuracy is None if not present or not parseable.

    Doctests:
        >>> parse_epoch_acc('epoch03_acc0.842.pt')
        (3, 0.842)
        >>> parse_epoch_acc('model_epoch12.pt')
        (12, None)
        >>> parse_epoch_acc('best-epoch_7-acc_0.912_ckpt.pt')
        (7, 0.912)
        >>> parse_epoch_acc('no_epoch_here.pt')
        Traceback (most recent call last):
        ...
        ValueError: could not find epoch number in filename: 'no_epoch_here.pt'
    """
    name = Path(filename).stem

    # Find epoch number after the token 'epoch' (case-insensitive).
    m = re.search(r"epoch[_-]?(\d+)|epoch(\d+)", name, flags=re.I)
    if not m:
        raise ValueError(f"could not find epoch number in filename: '{filename}'")

    # m.groups() contains either group 1 or 2 populated depending on which
    # alternative matched.
    epoch_str = next((g for g in m.groups() if g), None)
    epoch = int(epoch_str)

    # Try to find an accuracy float anywhere after 'acc' token.
    macc = re.search(r"acc[_-]?([0-9]*\.?[0-9]+)", name, flags=re.I)
    acc: Optional[float]
    if macc:
        try:
            acc = float(macc.group(1))
        except ValueError:
            acc = None
    else:
        acc = None

    return epoch, acc


class Timer:
    """Simple timer context manager that records elapsed seconds.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed)
    """

    def __init__(self) -> None:
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start = time.time()
        self.end = None
        self.elapsed = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.time()
        self.elapsed = self.end - (self.start or self.end)
        # don't suppress exceptions
        return False

