"""cache — Activation caching utilities

Disk-backed helpers to save/load activation tensors and small JSON metadata
for faster repeated analysis runs.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
import hashlib

import torch
import json
from typing import Any, Optional, Dict


def cache_key(model_name: str, layer_idx: int, split: str, tag: str) -> str:
    """Return a compact cache key for activations.

    The key encodes the model name, layer index, split and tag, but is
    hashed to keep filenames short and safe.
    """
    base = f"{model_name}_L{layer_idx}_{split}_{tag}"
    # short hash for uniqueness / filename safety
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    # keep a short readable prefix
    prefix = base.replace(" ", "_")
    # trim prefix to avoid overly long names
    if len(prefix) > 40:
        prefix = prefix[:40]
    return f"{prefix}_{h}"


def save_activations(tensor: torch.Tensor, out_dir: str = "activations_cache", key: Optional[str] = None, **meta) -> Path:
    """Save activations tensor to disk as a CPU .pt file with a JSON sidecar.

    Args:
        tensor: torch.Tensor containing activations (any device/shape)
        out_dir: directory to write cache files
        key: optional cache key (if None, meta should contain model_name/layer_idx/split/tag)
        **meta: additional metadata to write to the sidecar JSON

    Returns:
        Path to the saved .pt file
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if key is None:
        mn = meta.get("model_name", meta.get("model", "model"))
        li = meta.get("layer_idx", meta.get("layer", 0))
        sp = meta.get("split", meta.get("phase", "train"))
        tg = meta.get("tag", "anon")
        key = cache_key(str(mn), int(li), str(sp), str(tg))

    # filename base
    fname_base = key
    pt_path = out_dir_p / f"{fname_base}.pt"
    json_path = out_dir_p / f"{fname_base}.json"

    # Ensure tensor is on CPU and detached
    t_cpu = tensor.detach().cpu().clone()
    torch.save(t_cpu, str(pt_path))

    # enrich meta with save info
    meta_out: Dict[str, Any] = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}
    meta_out.update({"saved_at": time.time(), "filename": str(pt_path.name)})

    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(meta_out, jf, indent=2)

    return pt_path


def load_activations(in_dir: str = "activations_cache", key: Optional[str] = None) -> Optional[torch.Tensor]:
    """Load activations tensor from cache.

    If `key` is provided, attempts to find the most relevant file matching
    the key pattern. If `key` is None, returns the most recent .pt file in
    `in_dir`. Returns None when no matching file is found.
    """
    in_dir_p = Path(in_dir)
    if not in_dir_p.exists():
        return None

    candidates = list(in_dir_p.glob("*.pt"))
    if not candidates:
        return None

    if key is not None:
        # match files containing the key substring
        matches = [p for p in candidates if key in p.name]
        if not matches:
            return None
        # pick newest among matches
        chosen = max(matches, key=lambda p: p.stat().st_mtime)
    else:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)

    tensor = torch.load(str(chosen), map_location="cpu")
    # ensure tensor on CPU
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    else:
        # Unexpected object
        return None


def save_ranges(ranges: Dict[str, Any], out_dir: str = "activations_cache", key: Optional[str] = None) -> str:
    """Save a small JSON file containing calibration ranges (per-layer/input).

    Args:
        ranges: mapping-like object serializable to JSON
        out_dir: directory to store the JSON
        key: optional key to name the file (defaults to 'ranges')

    Returns:
        Path string to the saved JSON file.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    fname = f"{key or 'ranges'}.json"
    p = out_dir_p / fname
    with p.open("w", encoding="utf-8") as jf:
        json.dump(ranges, jf, indent=2)
    return str(p)


def load_ranges(key: Optional[str] = None, in_dir: str = "activations_cache") -> Optional[Dict[str, Any]]:
    """Load calibration ranges JSON file if present.

    Args:
        key: optional key used when saving (filename without .json)
        in_dir: directory to search

    Returns:
        Parsed dict or None if not found.
    """
    in_dir_p = Path(in_dir)
    if not in_dir_p.exists():
        return None
    fname = f"{key or 'ranges'}.json"
    p = in_dir_p / fname
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as jf:
        data = json.load(jf)
    return data


if __name__ == "__main__":
    # Quick smoke test: save and load a small tensor
    import torch as _torch

    t = _torch.randn(4, 8)
    key = cache_key("toy", 3, "val", "test")
    p = save_activations(t, out_dir="activations_cache", key=key, model_name="toy", layer_idx=3, split="val", tag="test")
    print("saved to", p)
    loaded = load_activations(in_dir="activations_cache", key=key)
    print("loaded shape", None if loaded is None else loaded.shape)
    assert loaded is not None and loaded.shape == t.shape


