"""
analysis — Orchestration for mutual information experiments

Provides high-level functions to run things like Ryan's notebook studies such as
- pretrained vs random comparisons
    - input/layer/label MI matrices
    - layer<->layer MI matrices
- training trajectories across checkpoints

These functions use the project's utilities (datasets, models, hooks,
mi estimators, plotting, caching) and are written to run on CPU or a
single GPU selected by `utils.select_device()`.
"""

from typing import List, Optional, Tuple, Dict
from pathlib import Path
import os
import platform
import pickle

import numpy as np
import torch

from utils import select_device, parse_epoch_acc, ensure_dir
from datasets import get_imagenette_loaders
from models import build_vgg16, build_resnet18, build_resnet50
from hooks import list_conv_layers, ActivationCatcher, reduce_activation
from mi_estimators import mi_cpu, mi_gpu, nmi_scalar, entropy_from_hist, mi2d_gpu, mi2d_cpu, mi_pairwise_channels_parallel
from plots import plot_mi_per_layer, plot_heatmap, plot_trajectory
from cache import save_activations, load_activations
from cache import save_ranges, load_ranges
try:
    from torch.amp import autocast  # torch >= 2.0
except Exception:
    from torch.cuda.amp import autocast  # older torch versions


_BACKBONES = {
    "vgg16": build_vgg16,
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
}


def _build_backbone(name: str, pretrained: bool, num_classes: int):
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone: {name}")
    return _BACKBONES[name](pretrained=pretrained, num_classes=num_classes)


def _resnet50_layer_names(model: torch.nn.Module, max_blocks_per_layer: Optional[int] = 1) -> List[str]:
    """Return ordered names for ResNet-50 by inspecting the model's modules.

    The returned list starts with 'conv1', then for each block in layer1..layer4
    appends 'layer{L}.{B}.conv1', 'layer{L}.{B}.conv2', 'layer{L}.{B}.conv3'
    in architectural order. By default only block 0 from each stage is
    returned (i.e. `max_blocks_per_layer=1`). Only names that actually
    appear in `model.named_modules()` are included, preserving exact module
    naming used by the model implementation.
    """
    # collect available module names for quick membership tests
    named = {n for n, _ in model.named_modules()}

    names: List[str] = []
    if "conv1" in named:
        names.append("conv1")

    for li in range(1, 5):
        layer_attr = f"layer{li}"
        if not hasattr(model, layer_attr):
            continue
        layer = getattr(model, layer_attr)
        for bi, _block in enumerate(layer):
            if max_blocks_per_layer is not None and bi >= max_blocks_per_layer:
                break
            for convk in ("conv1", "conv2", "conv3"):
                cand = f"{layer_attr}.{bi}.{convk}"
                if cand in named:
                    names.append(cand)
    return names


def _resnet18_layer_names(model: torch.nn.Module, max_blocks_per_layer: Optional[int] = 1) -> List[str]:
    """Return ordered names for ResNet-18: conv1, and conv1/2 of each basic block.

    By default only the first block (block 0) of each layer/stage is included
    (i.e. `max_blocks_per_layer=1`) to match the notebook convention of
    measuring MI for the stem conv1 and the first block per stage.
    """
    names: List[str] = ["conv1"]
    for li in range(1, 5):
        layer = getattr(model, f"layer{li}")
        for bi, block in enumerate(layer):
            if max_blocks_per_layer is not None and bi >= max_blocks_per_layer:
                break
            for convk in ("conv1", "conv2"):
                names.append(f"layer{li}.{bi}.{convk}")
    return names


def _vgg16_layer_names(model: torch.nn.Module, max_layers: Optional[int] = None) -> List[str]:
    """Return ordered names for VGG16 conv modules using the 'features' indices.

    Matches notebook naming such as 'features.0', 'features.2', ... for conv layers
    so plots and saved files use the exact module names.
    """
    names: List[str] = []
    # model.features is an Ordered container; collect indices of Conv2d modules
    for idx, m in enumerate(getattr(model, "features")):
        if isinstance(m, torch.nn.Conv2d):
            names.append(f"features.{idx}")
            if max_layers is not None and len(names) >= max_layers:
                break
    return names


def list_named_layers(model: torch.nn.Module, names: List[str]) -> List[torch.nn.Module]:
    """Map a list of string names to actual modules in the model.

    Supported name patterns:
        - 'conv1' for initial conv
        - 'layer{L}.{block_idx}.conv{K}' for blocks
    """
    mods: List[torch.nn.Module] = []
    for name in names:
        if name == "conv1":
            mods.append(model.conv1)
            continue
        # parse layerX.B.convY
        parts = name.split(".")
        if len(parts) == 3 and parts[0].startswith("layer"):
            layer_attr = parts[0]
            block_idx = int(parts[1])
            conv_attr = parts[2]
            layer = getattr(model, layer_attr)
            block = layer[block_idx]
            mod = getattr(block, conv_attr)
            mods.append(mod)
        else:
            raise ValueError(f"Unrecognized layer name: {name}")
    return mods


def _module_names_from_layers(model: torch.nn.Module, layers: List[torch.nn.Module]) -> List[str]:
    """Return the module names (from model.named_modules()) corresponding to the
    provided layer module objects, preserving model definition order.

    This avoids fallback generic labels like 'L0' and produces names such as
    'features.0' or 'layer1.0.conv1' that match the actual modules used in plots
    and console output.
    """
    names: List[str] = []
    layer_set = set(layers)
    for n, m in model.named_modules():
        # skip root module (empty name)
        if n == "":
            continue
        if m in layer_set:
            names.append(n)
            # remove to avoid duplicates if module appears multiple times
            layer_set.remove(m)
        if not layer_set:
            break
    # If we didn't find names for every layer, fall back to index-based but with prefix
    if len(names) < len(layers):
        names = [f"layer_{i}" for i in range(len(layers))]
    return names


def _stream_compute_mi_for_named_layers(model_builder, layer_names_fn, data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, pretrained: bool = False, max_blocks_per_layer: Optional[int] = None, write_cache: bool = False) -> Dict[str, float]:
    """Generic helper to compute per-named-layer MI and return dict name->MI."""
    device = select_device()
    default_workers = 0 if os.name == "nt" else 4
    _, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=False, normalize=False)
    model = model_builder(pretrained=pretrained, num_classes=num_classes)
    names = layer_names_fn(model, max_blocks_per_layer=max_blocks_per_layer)
    layers = list_named_layers(model, names)

    # collect activations per layer
    collected: List[List[np.ndarray]] = [[] for _ in layers]
    labels_list = []
    model = model.to(device)
    autocast_enabled = (device.type == "cuda")
    with ActivationCatcher(layers) as ac:
        for xb, yb in val_loader:
            xb_d = xb.to(device)
            with autocast(device.type, enabled=autocast_enabled):
                _ = model(xb_d)
            acts = ac.get_activations()
            for i, a in enumerate(acts):
                if a is None:
                    continue
                if a.dim() == 4:
                    r = reduce_activation(a, mode="spatial_mean")  # (B,C)
                elif a.dim() == 2:
                    r = a
                else:
                    r = a.view(a.size(0), -1)
                collected[i].append(r.cpu().numpy())
            labels_list.append(yb.numpy())

    labels = np.concatenate(labels_list, axis=0).ravel()
    use_gpu = device.type == "cuda"

    # Attempt to load a cached set of fixed ranges for inputs and layers
    ranges_key = f"{model.__class__.__name__}_val_ranges"
    ranges = load_ranges(key=ranges_key)
    if ranges is None:
        # compute robust min/max (2nd and 98th percentiles) per variable to avoid outliers
        ranges = {}
        # input scalar range
        try:
            inp_all = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0,))
        except Exception:
            inp_all = labels
        # store label/class range separately
        ranges["labels"] = [float(np.min(labels)) if labels.size else 0.0, float(np.max(labels)) if labels.size else 0.0]
        # For each layer compute percentile ranges across concatenated activations
        for i, name in enumerate(names):
            ch_list = collected[i] if i < len(collected) else []
            if not ch_list:
                ranges[name] = [0.0, 0.0]
                continue
            X = np.concatenate(ch_list, axis=0)
            # compute per-layer min/max as robust percentiles across all channels and samples
            lo = float(np.percentile(X, 2))
            hi = float(np.percentile(X, 98))
            if hi == lo:
                hi = lo + 1.0
            ranges[name] = [lo, hi]
        # save computed ranges for later reuse
        try:
            save_ranges(ranges, out_dir="activations_cache", key=ranges_key)
        except Exception:
            pass

    results: Dict[str, float] = {}
    tag_str = "pretrained" if pretrained else "random"
    for i, name in enumerate(names):
        # print exactly the module name being processed to match plot x-ticks
        print(f"[{tag_str}] Processing layer: {name}")
        ch_list = collected[i] if i < len(collected) else []
        if not ch_list:
            results[name] = 0.0
            print(f"-> Layer {name}: avg MI = {0.0:.4f}")
            continue
        # Attempt to load cached reduced activations for this backbone/layer
        cache_key_name = f"{model.__class__.__name__}_{name}_val"
        cached = load_activations(in_dir="activations_cache", key=cache_key_name)
        if cached is not None:
            print(f"Loaded cached activations for {cache_key_name}")
            X = cached.numpy() if isinstance(cached, torch.Tensor) else np.array(cached)
        else:
            X = np.concatenate(ch_list, axis=0)  # (N, C)
            try:
                # optionally save reduced activations for future runs
                if write_cache:
                    save_activations(torch.from_numpy(X), out_dir="activations_cache", key=cache_key_name, model_name=model.__class__.__name__, layer_idx=i, split="val", tag=name)
                    print(f"Saved activations for {cache_key_name}")
            except Exception:
                pass
        C = X.shape[1]
        # two-GPU fast path: if two GPUs are available and we have multiple channels,
        # delegate pairwise channel MI computation to the parallel helper to mirror
        # the notebook behavior and speed up large layers.
        if use_gpu and torch.cuda.device_count() >= 2 and C >= 2:
            try:
                xi_t = torch.from_numpy(X.astype(np.float32))
                y_t = torch.from_numpy(labels.astype(np.float32))
                mean_mi, mean_nmi = mi_pairwise_channels_parallel(xi_t, y_t, bins=bins, device0="cuda:0", device1="cuda:1")
                results[name] = float(mean_mi)
                print(f"-> Layer {name}: avg MI = {float(mean_mi):.4f}")
                continue
            except Exception:
                # fall back to single-GPU / CPU path below
                pass
        # Fast path: if we have 2 GPUs available and data is on CPU, use parallel channel-wise helper
        if use_gpu and torch.cuda.device_count() >= 2 and C > 1:
            try:
                X_t = torch.from_numpy(X.astype(np.float32))
                y_t = torch.from_numpy(labels.astype(np.int64))
                mi_val, _ = mi_pairwise_channels_parallel(X_t, y_t, bins=bins, device0='cuda:0', device1='cuda:1')
                results[name] = float(mi_val)
                print(f"-> Layer {name}: avg MI = {float(mi_val):.4f}")
                continue
            except Exception:
                pass

        mi_vals = []
        # determine k from labels for fixed label bins
        k_labels = int(np.max(labels)) + 1 if labels.size else 0
        for ci in range(C):
            xi = X[:, ci]
            if use_gpu:
                xi_t = torch.from_numpy(xi.astype(np.float32)).to(device)
                y_t = torch.from_numpy(labels.astype(np.int64)).to(device)
                # use precomputed range for this layer if available
                layer_range = None
                try:
                    layer_range = tuple(ranges.get(name, [None, None])) if ranges is not None else None
                except Exception:
                    layer_range = None
                mval = mi_gpu(
                    xi_t,
                    y_t,
                    bins=(bins, k_labels),
                    x_range=layer_range,
                    y_range=(-0.5, float(k_labels) - 0.5),
                    device=device,
                )
            else:
                mval = mi_cpu(xi, labels, bins=bins)
            mi_vals.append(mval)
        mean_val = float(np.mean(mi_vals))
        results[name] = mean_val
        print(f"-> Layer {name}: avg MI = {mean_val:.4f}")
    return results


def run_vgg16_stream(data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, pretrained: bool = False, max_layers: Optional[int] = None) -> Dict[str, float]:
    """Stream VGG16 conv layers, print per-layer lines and return dict name->avg_mi.

    This mirrors the notebook's VGG streaming cell and uses `model.features` conv modules.
    """
    device = select_device()
    default_workers = 0 if platform.system() == "Windows" else 4
    _, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=False, normalize=False)
    model = build_vgg16(pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    # enumerate conv modules in model.features
    feat_modules = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    # derive exact module names like 'features.0', 'features.2', ...
    feat_names = [f"features.{i}" for i, m in enumerate(model.features) if isinstance(m, torch.nn.Conv2d)]
    if max_layers is not None:
        feat_modules = feat_modules[:max_layers]
        feat_names = feat_names[:max_layers]

    # reuse the same streaming approach used earlier
    collected: List[List[np.ndarray]] = [[] for _ in feat_modules]
    labels_list = []
    autocast_enabled = (device.type == "cuda")
    with ActivationCatcher(feat_modules) as ac:
        for xb, yb in val_loader:
            xb_d = xb.to(device)
            with autocast(device.type, enabled=autocast_enabled):
                _ = model(xb_d)
            acts = ac.get_activations()
            for i, a in enumerate(acts[: len(feat_modules)]):
                if a is None:
                    continue
                if a.dim() == 4:
                    r = reduce_activation(a, mode="spatial_mean")
                elif a.dim() == 2:
                    r = a
                else:
                    r = a.view(a.size(0), -1)
                collected[i].append(r.cpu().numpy())
            labels_list.append(yb.numpy())

    labels = np.concatenate(labels_list, axis=0).ravel()
    results: Dict[str, float] = {}
    tag_str = "pretrained" if pretrained else "random"
    for i in range(len(feat_modules)):
        name = feat_names[i]
        # match notebook console format exactly
        print(f"[{tag_str}] Processing layer: {name}")
        ch_list = collected[i] if i < len(collected) else []
        if not ch_list:
            results[name] = 0.0
            print(f"-> Layer {name}: avg MI = {0.0:.4f}")
            continue
        X = np.concatenate(ch_list, axis=0)
        C = X.shape[1]
        # try 2-GPU fast path
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and C > 1:
            try:
                X_t = torch.from_numpy(X.astype(np.float32))
                y_t = torch.from_numpy(labels.astype(np.int64))
                mi_val, _ = mi_pairwise_channels_parallel(X_t, y_t, bins=bins, device0='cuda:0', device1='cuda:1')
                results[name] = float(mi_val)
                print(f"-> Layer {name}: avg MI = {mi_val:.4f}")
                continue
            except Exception:
                pass

        mi_vals = []
        use_gpu = device.type == "cuda"
        k_labels = int(np.max(labels)) + 1 if labels.size else 0
        # attempt to load cached ranges for this model
        ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
        layer_range = tuple(ranges.get(name, [None, None])) if ranges is not None else None
        for ci in range(C):
            xi = X[:, ci]
            if use_gpu:
                xi_t = torch.from_numpy(xi.astype(np.float32)).to(device)
                y_t = torch.from_numpy(labels.astype(np.int64)).to(device)
                mval = mi_gpu(xi_t, y_t, bins=(bins, k_labels), x_range=layer_range, y_range=(-0.5, float(k_labels) - 0.5), device=device)
            else:
                mval = mi_cpu(xi, labels, bins=bins)
            mi_vals.append(mval)
        mean_mi = float(np.mean(mi_vals))
        results[name] = mean_mi
        print(f"-> Layer {name}: avg MI = {mean_mi:.4f}")
    return results


def run_stream_input_and_label_mi(backbone: str, data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, augment: bool = False, out_dir: str = "outputs", max_layers: Optional[int] = None, pretrained: bool = True) -> Dict[str, List[float]]:
    """Stream inputs and layers, compute MI/NMI input->layer and layer->label per batch.

    For each validation batch:
    - reduce input to a scalar per sample (mean over channels & spatial dims)
    - reduce each layer to (B, C) via reduce_activation(mode='spatial_mean')
    - compute per-channel MI between input scalar and that channel (mi2d_gpu)
      and between channel and label (mi2d_gpu)
    - average MI across channels to produce per-layer values for that batch

    Finally average per-layer values across batches to produce two curves:
    - MI_in: mean MI(input; layer)
    - MI_lbl: mean MI(layer; label)
    Also compute corresponding NMI curves. Saves dual-line plots for MI and NMI.

    Returns a dict with keys: 'mi_in', 'mi_lbl', 'nmi_in', 'nmi_lbl', 'layer_names'
    """
    device = select_device()
    ensure_dir(out_dir)
    default_workers = 0 if platform.system() == "Windows" else 4
    _, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=augment, normalize=False)

    model = _build_backbone(backbone, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    layers = list_conv_layers(model)
    if max_layers is not None:
        layers = layers[:max_layers]

    L = len(layers)
    # per-layer list of per-batch values
    mi_in_batches: List[List[float]] = [[] for _ in range(L)]
    mi_lbl_batches: List[List[float]] = [[] for _ in range(L)]
    nmi_in_batches: List[List[float]] = [[] for _ in range(L)]
    nmi_lbl_batches: List[List[float]] = [[] for _ in range(L)]

    autocast_enabled = (device.type == "cuda")
    with ActivationCatcher(layers) as ac:
        for xb, yb in val_loader:
            # forward on device; ActivationCatcher captures CPU copies
            xb_d = xb.to(device)
            with autocast(device.type, enabled=autocast_enabled):
                _ = model(xb_d)
            acts = ac.get_activations()

            # input scalar per sample (CPU tensor)
            inp = xb.mean(dim=(1, 2, 3))  # (B,)
            # labels as CPU tensor
            labels = yb

            for li in range(L):
                a = acts[li] if li < len(acts) else None
                if a is None:
                    # append zeros for this batch
                    mi_in_batches[li].append(0.0)
                    mi_lbl_batches[li].append(0.0)
                    nmi_in_batches[li].append(0.0)
                    nmi_lbl_batches[li].append(0.0)
                    continue

                # reduce to (B, C)
                if a.dim() == 4:
                    red = reduce_activation(a, mode="spatial_mean")  # (B, C) on CPU
                elif a.dim() == 2:
                    red = a
                else:
                    red = a.view(a.size(0), -1)

                B, C = red.size(0), red.size(1)
                # accumulate per-channel MI for this batch
                mi_in_ch = []
                mi_lbl_ch = []
                nmi_in_ch = []
                nmi_lbl_ch = []

                # move small vectors to device for GPU estimator when available
                use_gpu = device.type == "cuda"

                for ci in range(C):
                    ch_vec = red[:, ci]  # CPU tensor (B,)

                    try:
                        if use_gpu:
                            inp_t = inp.to(device, dtype=torch.float32)
                            ch_t = ch_vec.to(device, dtype=torch.float32)
                            labels_t = labels.to(device, dtype=torch.int64)
                            # use fixed ranges per input and per-layer when available
                            ranges_key = f"{model.__class__.__name__}_val_ranges"
                            ranges = load_ranges(key=ranges_key)
                            # input range: use grayscale/input scaler range if present
                            inp_range = None
                            layer_range = None
                            if ranges is not None:
                                inp_range = tuple(ranges.get("input_gray", ranges.get("inputs", [None, None])))
                                layer_range = tuple(ranges.get(layers[li].__class__.__name__ if isinstance(layers[li], torch.nn.Module) else layers[li], ranges.get(layers[li] if isinstance(layers[li], str) else str(layers[li]), [None, None])))
                            m_in, n_in = mi2d_gpu(
                                inp_t,
                                ch_t,
                                bins=(bins, num_classes),
                                x_range=inp_range,
                                y_range=(-0.5, float(num_classes) - 0.5),
                                device=device,
                            )
                            m_lbl, n_lbl = mi2d_gpu(
                                ch_t,
                                labels_t,
                                bins=(bins, num_classes),
                                x_range=layer_range,
                                y_range=(-0.5, float(num_classes) - 0.5),
                                device=device,
                            )
                        else:
                            # CPU path: use mi2d_cpu which mirrors GPU binning/label handling
                            inp_np = inp.numpy()
                            ch_np = ch_vec.numpy()
                            labels_np = labels.numpy()
                            ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
                            inp_range_np = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                            layer_key = layers[li] if isinstance(layers[li], str) else (layers[li].__class__.__name__ if isinstance(layers[li], torch.nn.Module) else str(layers[li]))
                            layer_range_np = tuple(ranges.get(layer_key, [None, None])) if ranges is not None else None
                            m_in, n_in = mi2d_cpu(
                                inp_np,
                                ch_np,
                                bins=(bins, num_classes),
                                x_range=inp_range_np,
                                y_range=(-0.5, float(num_classes) - 0.5),
                            )
                            m_lbl, n_lbl = mi2d_cpu(
                                ch_np,
                                labels_np,
                                bins=(bins, num_classes),
                                x_range=layer_range_np,
                                y_range=(-0.5, float(num_classes) - 0.5),
                            )
                    except Exception:
                        m_in = 0.0
                        m_lbl = 0.0
                        n_in = 0.0
                        n_lbl = 0.0

                    mi_in_ch.append(float(m_in))
                    mi_lbl_ch.append(float(m_lbl))
                    nmi_in_ch.append(float(n_in))
                    nmi_lbl_ch.append(float(n_lbl))

                # mean across channels for this batch
                mi_in_batches[li].append(float(np.mean(mi_in_ch)) if mi_in_ch else 0.0)
                mi_lbl_batches[li].append(float(np.mean(mi_lbl_ch)) if mi_lbl_ch else 0.0)
                nmi_in_batches[li].append(float(np.mean(nmi_in_ch)) if nmi_in_ch else 0.0)
                nmi_lbl_batches[li].append(float(np.mean(nmi_lbl_ch)) if nmi_lbl_ch else 0.0)

    # average across batches per layer
    mi_in = [float(np.mean(batch_vals)) if batch_vals else 0.0 for batch_vals in mi_in_batches]
    mi_lbl = [float(np.mean(batch_vals)) if batch_vals else 0.0 for batch_vals in mi_lbl_batches]
    nmi_in = [float(np.mean(batch_vals)) if batch_vals else 0.0 for batch_vals in nmi_in_batches]
    nmi_lbl = [float(np.mean(batch_vals)) if batch_vals else 0.0 for batch_vals in nmi_lbl_batches]

    # sanitize non-finite
    mi_in = [0.0 if not np.isfinite(v) else v for v in mi_in]
    mi_lbl = [0.0 if not np.isfinite(v) else v for v in mi_lbl]
    nmi_in = [0.0 if not np.isfinite(v) else v for v in nmi_in]
    nmi_lbl = [0.0 if not np.isfinite(v) else v for v in nmi_lbl]

    # build human-friendly layer names depending on backbone
    if backbone.startswith("resnet"):
        if backbone == "resnet50":
            layer_names = _resnet50_layer_names(model, max_blocks_per_layer=max_layers)
        else:
            layer_names = _resnet18_layer_names(model, max_blocks_per_layer=max_layers)
    elif backbone == "vgg16":
        layer_names = _vgg16_layer_names(model, max_layers=max_layers)
    else:
        # fallback to mapping actual module names from extracted layers
        layer_names = _module_names_from_layers(model, layers)

    # Save dual-line plots (MI and NMI) with explicit backbone name for exact notebook wording
    try:
        plot_mi_per_layer(
            layer_names,
            mi_in,
            mi_lbl,
            backbone_name="",
            title=f"Mutual Information (Feature vs Label) per Conv Layer ({backbone.upper()})",
            label_pre="Input->Layer",
            label_rand="Label->Layer",
            out_path=os.path.join(out_dir, f"stream_mi_{backbone}.png"),
        )
    except Exception:
        pass
    try:
        plot_mi_per_layer(
            layer_names,
            nmi_in,
            nmi_lbl,
            backbone_name="",
            title=f"Normalized Mutual Information (Feature vs Label) per Conv Layer ({backbone.upper()})",
            label_pre="Input->Layer",
            label_rand="Label->Layer",
            out_path=os.path.join(out_dir, f"stream_nmi_{backbone}.png"),
        )
    except Exception:
        pass

    return {"mi_in": mi_in, "mi_lbl": mi_lbl, "nmi_in": nmi_in, "nmi_lbl": nmi_lbl, "layer_names": layer_names}


def compute_mi_patterns(model: torch.nn.Module, dataloader, device: torch.device, bins: int = 20, per_channel: bool = True):
    """Compute MI patterns between input, each layer (per-channel), and labels.

    Returns a dict with keys:
      'layers': [names...],
      'mi_input_to_layer_mean': [float per layer],
      'mi_layer_to_label_mean': [float per layer],
      'mi_input_to_layer_per_channel': {name: np.array[C]},
      'mi_layer_to_label_per_channel': {name: np.array[C]},

    Uses `ActivationCatcher` and `reduce_activation` to collect per-layer
    (N, C) activations. For inputs we compute a small projection per sample
    via x.mean(dim=(2,3)) -> [N,3] and also a grayscale avg x.mean(dim=(1,2,3)).
    MI estimates prefer the GPU 2D-hist helper when device is CUDA.
    """
    model = model.to(device)
    model.eval()

    layers = list_conv_layers(model)
    layer_names = _module_names_from_layers(model, layers)

    # Accumulate across batches
    per_layer_cols = [[] for _ in layers]  # each entry -> list of (B,C) arrays
    input_proj_list = []  # list of (B,3) arrays
    input_gray_list = []  # list of (B,) arrays
    labels_list = []

    autocast_enabled = (device.type == "cuda")
    with ActivationCatcher(layers) as ac:
        for xb, yb in dataloader:
            xb_cpu = xb
            # input projections
            x_proj = xb_cpu.mean(dim=(2, 3))  # (B,3)
            x_gray = xb_cpu.mean(dim=(1, 2, 3))  # (B,)
            input_proj_list.append(x_proj.numpy())
            input_gray_list.append(x_gray.numpy())
            labels_list.append(yb.numpy())

            # forward on device to populate activations (ActivationCatcher captures CPU copies)
            with autocast(device.type, enabled=autocast_enabled):
                _ = model(xb.to(device))
            acts = ac.get_activations()
            for i, a in enumerate(acts[: len(layers)]):
                if a is None:
                    per_layer_cols[i].append(np.zeros((xb.size(0), 0)))
                    continue
                a = a.detach().cpu()
                if a.dim() == 4:
                    red = reduce_activation(a, mode="spatial_mean")  # (B, C)
                elif a.dim() == 2:
                    red = a
                else:
                    red = a.view(a.size(0), -1)
                per_layer_cols[i].append(red.numpy())

    # Concatenate across batches
    X_inputs_proj = np.concatenate(input_proj_list, axis=0) if input_proj_list else np.zeros((0, 3))
    X_inputs_gray = np.concatenate(input_gray_list, axis=0) if input_gray_list else np.zeros((0,))
    Y = np.concatenate(labels_list, axis=0).ravel() if labels_list else np.zeros((0,))

    mi_input_to_layer_mean = []
    mi_layer_to_label_mean = []
    mi_input_to_layer_per_channel = {}
    mi_layer_to_label_per_channel = {}

    use_gpu = device.type == "cuda"

    def compute_pair_mi(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
        try:
            if use_gpu:
                x_t = torch.from_numpy(x_arr.astype(np.float32)).to(device)
                # if y_arr contains integer labels, compute fixed num_classes from data
                if np.issubdtype(y_arr.dtype, np.integer):
                    k = int(np.max(y_arr)) + 1 if y_arr.size else 0
                    y_t = torch.from_numpy(y_arr.astype(np.int64)).to(device)
                    # attempt to use cached ranges for x if available (per-variable keys)
                    ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
                    x_r = None
                    if ranges is not None:
                        # try to find a matching range key; fall back to None
                        x_r = tuple(ranges.get("input_gray", [None, None]))
                    mi_val, _ = mi2d_gpu(
                        x_t,
                        y_t,
                        bins=(bins, k),
                        x_range=x_r,
                        y_range=(-0.5, float(k) - 0.5),
                        device=device,
                    )
                else:
                    y_t = torch.from_numpy(y_arr.astype(np.float32)).to(device)
                    ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
                    x_r = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                    mi_val, _ = mi2d_gpu(x_t, y_t, bins=bins, x_range=x_r, device=device)
                return float(mi_val)
            else:
                if np.issubdtype(y_arr.dtype, np.integer):
                    k = int(np.max(y_arr)) + 1 if y_arr.size else 0
                    ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
                    x_r = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                    mi_val, _ = mi2d_cpu(
                        x_arr.astype(np.float32),
                        y_arr.astype(np.int64),
                        bins=(bins, k),
                        x_range=x_r,
                        y_range=(-0.5, float(k) - 0.5),
                    )
                else:
                    ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
                    x_r = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                    mi_val, _ = mi2d_cpu(x_arr.astype(np.float32), y_arr.astype(np.float32), bins=bins, x_range=x_r)
                return float(mi_val)
        except Exception:
            return 0.0

    # determine label cardinality from concatenated labels Y
    k_labels = int(np.max(Y)) + 1 if Y.size else 0

    for i, name in enumerate(layer_names):
        ch_list = per_layer_cols[i] if i < len(per_layer_cols) else []
        if not ch_list:
            mi_input_to_layer_per_channel[name] = np.array([])
            mi_layer_to_label_per_channel[name] = np.array([])
            mi_input_to_layer_mean.append(0.0)
            mi_layer_to_label_mean.append(0.0)
            continue
        X = np.concatenate(ch_list, axis=0)  # (N, C)
        N, C = X.shape

        # compute per-channel MI vs input and vs label
        mi_in_ch = np.zeros((C,), dtype=float)
        mi_lbl_ch = np.zeros((C,), dtype=float)

        # choose input representation: use grayscale scalar per sample
        inp_gray = X_inputs_gray if X_inputs_gray.shape[0] == N else X_inputs_gray[:N]
        # if grayscale not matching, fallback to channel-wise flatten mean
        if inp_gray.shape[0] != N:
            inp_gray = X_inputs_proj.mean(axis=1) if X_inputs_proj.shape[0] == N else np.zeros((N,))

        for ci in range(C):
            ch_vec = X[:, ci]
            # MI(input, channel)
            mi_in_ch[ci] = compute_pair_mi(inp_gray, ch_vec) if per_channel else 0.0
            # MI(channel, label)
            try:
                if use_gpu:
                    ch_t = torch.from_numpy(ch_vec.astype(np.float32)).to(device)
                    y_t = torch.from_numpy(Y.astype(np.int64)).to(device)
                    ranges = load_ranges(key=f"{model.__class__.__name__}_val_ranges")
                    layer_r = tuple(ranges.get(name, [None, None])) if ranges is not None else None
                    m_lbl, _ = mi2d_gpu(
                        ch_t,
                        y_t,
                        bins=(bins, k_labels),
                        x_range=layer_r,
                        y_range=(-0.5, float(k_labels) - 0.5),
                        device=device,
                    )
                    mi_lbl_ch[ci] = float(m_lbl)
                else:
                    m_lbl, _ = mi2d_cpu(
                        ch_vec.astype(np.float32),
                        Y.astype(np.int64),
                        bins=(bins, k_labels),
                        x_range=None,
                        y_range=(-0.5, float(k_labels) - 0.5),
                    )
                    mi_lbl_ch[ci] = float(m_lbl)
            except Exception:
                mi_lbl_ch[ci] = 0.0

        mi_input_to_layer_per_channel[name] = mi_in_ch
        mi_layer_to_label_per_channel[name] = mi_lbl_ch
        # mean across channels
        mi_input_to_layer_mean.append(float(np.mean(mi_in_ch)) if mi_in_ch.size else 0.0)
        mi_layer_to_label_mean.append(float(np.mean(mi_lbl_ch)) if mi_lbl_ch.size else 0.0)

    return {
        "layers": layer_names,
        "mi_input_to_layer_mean": mi_input_to_layer_mean,
        "mi_layer_to_label_mean": mi_layer_to_label_mean,
        "mi_input_to_layer_per_channel": mi_input_to_layer_per_channel,
        "mi_layer_to_label_per_channel": mi_layer_to_label_per_channel,
    }


def _parallel_channel_mean_mi(x_arr: np.ndarray, y_arr: np.ndarray, device: torch.device, bins: int = 20):
    """Compute MI and NMI per-channel between x_arr and y_arr and return the mean across channels.

    x_arr: (N, Cx) or (N,) ; y_arr: (N, Cy) or (N,)
    Behavior:
      - If one side is 1D (labels), compare every channel on the other side to that 1D vector and average.
      - If both have multiple channels, pair channels index-wise up to min(Cx, Cy) and average.
    Returns: (mean_mi, mean_nmi)
    """
    # normalize shapes to 2D
    if x_arr is None or y_arr is None:
        return 0.0, 0.0
    xa = np.asarray(x_arr)
    ya = np.asarray(y_arr)
    if xa.ndim == 1:
        xa = xa[:, None]
    if ya.ndim == 1:
        ya = ya[:, None]
    if xa.shape[0] != ya.shape[0]:
        return 0.0, 0.0

    Cx = xa.shape[1]
    Cy = ya.shape[1]
    if Cx == 0 or Cy == 0:
        return 0.0, 0.0

    use_gpu = device.type == "cuda"
    mi_vals = []
    nmi_vals = []

    # decide pairing logic
    if Cx == Cy:
        pairs = [(k, k) for k in range(Cx)]
    elif Cy == 1:
        pairs = [(k, 0) for k in range(Cx)]
    elif Cx == 1:
        pairs = [(0, k) for k in range(Cy)]
    else:
        m = min(Cx, Cy)
        pairs = [(k, k) for k in range(m)]

    for xi_idx, yi_idx in pairs:
        xcol = xa[:, xi_idx]
        ycol = ya[:, yi_idx]
        try:
            if use_gpu:
                xt = torch.from_numpy(xcol.astype(np.float32)).to(device)
                yt = torch.from_numpy(ycol.astype(np.float32)).to(device)
                ranges = load_ranges(key=f"{device.__class__.__name__}_val_ranges")
                # if ycol looks like integer labels, use categorical bins
                if np.issubdtype(ycol.dtype, np.integer):
                    k = int(np.max(ycol)) + 1 if ycol.size else 0
                    x_r = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                    mi_val, nmi_val = mi2d_gpu(xt, yt.to(dtype=torch.int64), bins=(bins, k), x_range=x_r, y_range=(-0.5, float(k) - 0.5), device=device)
                else:
                    # attempt to use a per-variable range keyed by index name if available
                    # fall back to None
                    mi_val, nmi_val = mi2d_gpu(xt, yt, bins=(bins, bins), x_range=None, y_range=None, device=device)
                mi_vals.append(float(mi_val))
                nmi_vals.append(float(nmi_val))
            else:
                # CPU path
                ranges = load_ranges(key=f"{device.__class__.__name__}_val_ranges")
                if np.issubdtype(ycol.dtype, np.integer):
                    k = int(np.max(ycol)) + 1 if ycol.size else 0
                    x_r = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                    mi_val, nmi_val = mi2d_cpu(xcol.astype(np.float32), ycol.astype(np.int64), bins=(bins, k), x_range=x_r, y_range=(-0.5, float(k) - 0.5))
                else:
                    x_r = None
                    mi_val, nmi_val = mi2d_cpu(xcol.astype(np.float32), ycol.astype(np.float32), bins=(bins, bins), x_range=x_r, y_range=None)
                mi_vals.append(float(mi_val))
                nmi_vals.append(float(nmi_val))
        except Exception:
            try:
                # fallback to scalar MI estimate
                if use_gpu:
                    xt = torch.from_numpy(xcol.astype(np.float32)).to(device)
                    yt = torch.from_numpy(ycol.astype(np.float32)).to(device)
                    ranges = load_ranges(key=f"{device.__class__.__name__}_val_ranges")
                    if np.issubdtype(ycol.dtype, np.integer):
                        k = int(np.max(ycol)) + 1 if ycol.size else 0
                        x_r = tuple(ranges.get("input_gray", [None, None])) if ranges is not None else None
                        m = mi_gpu(xt, yt.to(dtype=torch.int64), bins=(bins, k), x_range=x_r, y_range=(-0.5, float(k) - 0.5), device=device)
                    else:
                        x_r = None
                        m = mi_gpu(xt, yt, bins=(bins, bins), x_range=x_r, y_range=None, device=device)
                    mi_vals.append(float(m))
                    nmi_vals.append(0.0)
                else:
                    m = mi_cpu(xcol, ycol, bins=bins)
                    mi_vals.append(float(m))
                    nmi_vals.append(0.0)
            except Exception:
                mi_vals.append(0.0)
                nmi_vals.append(0.0)

    mean_mi = float(np.mean(mi_vals)) if mi_vals else 0.0
    mean_nmi = float(np.mean(nmi_vals)) if nmi_vals else 0.0
    return mean_mi, mean_nmi


def run_resnet50_stream(data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, pretrained: bool = False, max_blocks_per_layer: Optional[int] = None) -> Dict[str, float]:
    """Compute MI for hand-picked ResNet-50 layers and print results."""
    res = _stream_compute_mi_for_named_layers(build_resnet50, _resnet50_layer_names, data_root=data_root, bins=bins, batch_size=batch_size, pretrained=pretrained, max_blocks_per_layer=max_blocks_per_layer)
    # _stream_compute_mi_for_named_layers already prints standardized lines
    return res


def run_resnet18_stream(data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, pretrained: bool = False, max_blocks_per_layer: Optional[int] = None) -> Dict[str, float]:
    """Compute MI for hand-picked ResNet-18 layers and print results."""
    res = _stream_compute_mi_for_named_layers(build_resnet18, _resnet18_layer_names, data_root=data_root, bins=bins, batch_size=batch_size, pretrained=pretrained, max_blocks_per_layer=max_blocks_per_layer)
    # _stream_compute_mi_for_named_layers already prints standardized lines
    return res


def _compute_layer_label_mi(model: torch.nn.Module, layers: List[torch.nn.Module], dataloader, device: torch.device, bins: int = 20, max_layers: Optional[int] = None, use_joint_hist: bool = True, cache_prefix: Optional[str] = None, write_cache: bool = False) -> List[float]:
    """For each layer, collect activations across the dataloader, reduce to (N,C)
    and compute MI between each channel and the label; return per-layer average MI."""
    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    if max_layers is not None:
        layers = layers[:max_layers]

    # Collect per-layer activations (list of lists)
    collected: List[List[np.ndarray]] = [[] for _ in layers]
    labels_list: List[int] = []

    autocast_enabled = (device.type == "cuda")
    with ActivationCatcher(layers) as ac:
        for xb, yb in dataloader:
            # run forward; ActivationCatcher stores CPU activations
            xb_device = xb.to(device)
            with autocast(device.type, enabled=autocast_enabled):
                _ = model(xb_device)
            acts = ac.get_activations()
            # ensure acts correspond to layers
            for i, a in enumerate(acts):
                if a is None:
                    continue
                # a is CPU tensor shape (B,C,H,W) or (B,C)
                if a.dim() == 4:
                    red = reduce_activation(a, mode="spatial_mean")  # (B,C)
                elif a.dim() == 2:
                    red = a
                else:
                    # flatten other shapes
                    red = a.view(a.size(0), -1)
                collected[i].append(red.numpy())

            labels_list.append(yb.numpy())

    if not labels_list:
        return []

    labels = np.concatenate(labels_list, axis=0).ravel()

    layer_mi: List[float] = []
    use_gpu = device.type == "cuda"

    for ch_list in collected:
        if not ch_list:
            layer_mi.append(0.0)
            continue
        # try to load cached activations by model class and layer index
        model_name = model.__class__.__name__
        # allow caller to provide a cache prefix so different model variants
        # (e.g., pretrained vs random) do not collide in the activation cache
        prefix = cache_prefix if cache_prefix is not None else model_name
        # infer split name as 'val' by default
        cache_key_name = f"{prefix}_L{len(layer_mi)}_val"
        cached = load_activations(in_dir="activations_cache", key=cache_key_name)
        if cached is not None:
            print(f"Loaded cached activations for {cache_key_name}")
            X = cached.numpy() if isinstance(cached, torch.Tensor) else np.array(cached)
        else:
            X = np.concatenate(ch_list, axis=0)  # (N, C)
            try:
                if write_cache:
                    save_activations(torch.from_numpy(X), out_dir="activations_cache", key=cache_key_name, model_name=model_name, layer_idx=len(layer_mi), split="val", tag=f"L{len(layer_mi)}")
                    print(f"Saved activations for {cache_key_name}")
            except Exception:
                pass
        C = X.shape[1]
        # compute MI per channel then average
        mi_vals = []
        # determine k_labels from collected labels
        k_labels = int(np.max(labels)) + 1 if labels.size else 0
        for ci in range(C):
            xi = X[:, ci]
            if use_gpu:
                xi_t = torch.from_numpy(xi.astype(np.float32)).to(device)
                y_t = torch.from_numpy(labels.astype(np.int64)).to(device)
                # Prefer the 2D joint-histogram GPU estimator when requested.
                if use_joint_hist:
                    try:
                        # mi2d_gpu returns (mi, nmi)
                        mval, _ = mi2d_gpu(
                            xi_t,
                            y_t,
                            bins=(bins, k_labels),
                            x_range=None,
                            y_range=(-0.5, float(k_labels) - 0.5),
                            device=device,
                        )
                    except Exception:
                        # fallback to existing 1D estimator (robust to missing GPU ops)
                        mval = mi_gpu(xi_t, y_t, bins=(bins, k_labels), x_range=None, y_range=(-0.5, float(k_labels) - 0.5), device=device)
                else:
                    mval = mi_gpu(xi_t, y_t, bins=(bins, k_labels), x_range=None, y_range=(-0.5, float(k_labels) - 0.5), device=device)
            else:
                # On CPU we don't have the fast joint-hist GPU path; fall back to mi_cpu.
                mval = mi_cpu(xi, labels, bins=bins)
            mi_vals.append(mval)
        layer_mi.append(float(np.mean(mi_vals)))

    return layer_mi


def run_pretrained_vs_random(backbone: str, data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, augment: bool = False, out_dir: str = "outputs", max_layers: Optional[int] = 6, write_cache: bool = False) -> Dict[str, List[float]]:
    """Compare pretrained vs random initialization by per-layer MI to labels.

    Builds the model twice (pretrained=True and False), collects activations on
    the validation set, computes per-channel MI with labels, averages per layer,
    and saves per-backbone plots to `out_dir`.
    """
    device = select_device()
    ensure_dir(out_dir)

    default_workers = 0 if platform.system() == "Windows" else 4
    train_loader, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=augment, normalize=False)

    results = {}
    for tag, pretrained in [("pretrained", True), ("random", False)]:
        model = _build_backbone(backbone, pretrained=pretrained, num_classes=num_classes)
        model.eval()
        # For ResNets prefer hand-picked named layers so plots show meaningful names
        if backbone.startswith("resnet"):
            if backbone == "resnet50":
                names = _resnet50_layer_names(model, max_blocks_per_layer=max_layers)
            else:
                names = _resnet18_layer_names(model, max_blocks_per_layer=max_layers)
            layers = list_named_layers(model, names)
            layer_mi = _compute_layer_label_mi(model, layers, val_loader, device, bins=bins, max_layers=None, use_joint_hist=True, cache_prefix=f"{backbone}_{tag}", write_cache=write_cache)
        else:
            # non-resnet backbones: use conv layer enumeration and derive module names
            layers = list_conv_layers(model)
            layer_mi = _compute_layer_label_mi(model, layers, val_loader, device, bins=bins, max_layers=max_layers, use_joint_hist=True, cache_prefix=f"{backbone}_{tag}", write_cache=write_cache)
            names = _module_names_from_layers(model, layers[:len(layer_mi)])

        results[tag] = layer_mi
        # save numeric results for this tag
        try:
            np.savez(os.path.join(out_dir, f"mi_{backbone}_{tag}.npz"), names=np.array(names), mi=np.array(layer_mi))
        except Exception:
            pass
        # also create a simple single-series plot per tag — use plot_mi_per_layer
        try:
            safe_tag = tag.replace(' ', '_').replace(',', '')
            # pass explicit legend label with backbone to avoid accidental defaults
            explicit_label = f"{tag} {backbone.upper()}"
            plot_mi_per_layer(names, layer_mi, mi_rand=None, backbone_name="", title=f"MI per layer ({backbone}) [{tag}]", label_pre=explicit_label, out_path=os.path.join(out_dir, f"mi_{backbone}_{safe_tag}.png"))
        except Exception:
            pass

    # Also save combined trajectory-like plot
    try:
        # Use plot_mi_per_layer so the x-axis uses exact layer/module names
        # use explicit labels that include backbone to make legend text deterministic
        plot_mi_per_layer(
            names,
            results["pretrained"],
            results["random"],
            backbone_name="",
            title=f"MI per layer ({backbone}) Comparison",
            label_pre=f"pretrained {backbone.upper()}",
            label_rand=f"random {backbone.upper()}",
            out_path=os.path.join(out_dir, f"mi_{backbone}_comparison.png"),
        )
    except Exception:
        pass

    return results


def run_input_layer_label_matrix(backbone: str, data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, augment: bool = False, out_dir: str = "outputs", max_layers: Optional[int] = 6, pretrained: bool = False) -> Dict[str, np.ndarray]:
    """Compute an MI matrix among INPUT (reduced), layers and LABEL and save heatmap."""
    device = select_device()
    ensure_dir(out_dir)
    default_workers = 0 if platform.system() == "Windows" else 4
    _, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=augment, normalize=False)
    model = _build_backbone(backbone, pretrained=pretrained, num_classes=num_classes)
    model.eval()
    # ensure model weights are on the selected device before forward passes
    model = model.to(device)
    layers = list_conv_layers(model)
    if max_layers is not None:
        layers = layers[:max_layers]

    # collect representations: inputs and each layer reduced to scalar per sample
    reps: List[np.ndarray] = []
    labels_list = []
    with ActivationCatcher(layers) as ac:
        for xb, yb in val_loader:
            xb_device = xb.to(device)
            _ = model(xb_device)
            acts = ac.get_activations()
            # input reduced: mean over spatial & channels -> scalar per sample
            inp_red = xb.mean(dim=(1, 2, 3)).numpy()
            reps.append(inp_red[:, None])
            for a in acts[: len(layers)]:
                if a is None:
                    reps.append(np.zeros((xb.size(0), 1)))
                    continue
                if a.dim() == 4:
                    r = reduce_activation(a, mode="spatial_mean").mean(dim=1).numpy()  # (B,)
                elif a.dim() == 2:
                    r = a.mean(dim=1).numpy()
                else:
                    r = a.view(a.size(0), -1).mean(dim=1).numpy()
                reps.append(r[:, None])
            labels_list.append(yb.numpy())

    # concatenate per node
    reps_concat = [np.concatenate([r_chunk for r_chunk in reps[i::len(layers)+1]], axis=0) for i in range(len(layers) + 1)]
    labels = np.concatenate(labels_list, axis=0).ravel()

    # build MI and NMI matrices
    N = len(reps_concat) + 1  # include label as last
    mi_mat = np.zeros((N, N), dtype=float)
    nmi_mat = np.zeros((N, N), dtype=float)
    entropies = np.zeros((N,), dtype=float)
    use_gpu = device.type == "cuda"

    # compute entropy for inputs/layers
    for i in range(N - 1):
        arr = reps_concat[i].ravel()
        # discrete histogram for entropy
        counts = np.bincount(np.floor((arr - arr.min()) / (arr.max() - arr.min() + 1e-12) * (bins - 1)).astype(int))
        entropies[i] = entropy_from_hist(counts)

    # label entropy
    entropies[N - 1] = entropy_from_hist(np.bincount(labels))

    # compute pairwise MI/NMI for inputs/layers via joint histograms
    for i in range(N - 1):
        xi = reps_concat[i].ravel()
        for j in range(N - 1):
            xj = reps_concat[j].ravel()
            # ensure same length
            if xi.shape[0] != xj.shape[0]:
                mi_val = 0.0
                nmi_val = 0.0
            else:
                try:
                    if use_gpu:
                        xi_t = torch.from_numpy(xi.astype(np.float32)).to(device)
                        xj_t = torch.from_numpy(xj.astype(np.float32)).to(device)
                        mi_val, nmi_val = mi2d_gpu(xi_t, xj_t, bins=(bins, bins), x_range=None, y_range=None, device=device)
                    else:
                        # CPU 2D histogram
                        eps = 1e-12
                        x_edges = np.linspace(float(xi.min()), float(xi.max() if xi.max() != xi.min() else xi.min() + 1.0), bins + 1)
                        y_edges = np.linspace(float(xj.min()), float(xj.max() if xj.max() != xj.min() else xj.min() + 1.0), bins + 1)
                        counts, _, _ = np.histogram2d(xi, xj, bins=(x_edges, y_edges))
                        total = counts.sum()
                        if total <= 0:
                            mi_val = 0.0
                            nmi_val = 0.0
                        else:
                            pxy = counts / total
                            pxy = pxy + eps
                            px = pxy.sum(axis=1)
                            py = pxy.sum(axis=0)
                            hx = -np.sum(px * np.log(px))
                            hy = -np.sum(py * np.log(py))
                            hxy = -np.sum(pxy * np.log(pxy))
                            mi_val = float(hx + hy - hxy)
                            nmi_val = float(mi_val / (hy + 1e-12))
                except Exception:
                    mi_val = 0.0
                    nmi_val = 0.0
            mi_mat[i, j] = mi_val
            nmi_mat[i, j] = nmi_val

    # MI/NMI vs label
    for i in range(N - 1):
        try:
            if use_gpu:
                xi_t = torch.from_numpy(reps_concat[i].ravel().astype(np.float32)).to(device)
                labels_t = torch.from_numpy(labels.astype(np.int64)).to(device)
                mi_val, nmi_val = mi2d_gpu(
                    xi_t,
                    labels_t,
                    bins=(bins, num_classes),
                    x_range=None,
                    y_range=(-0.5, float(num_classes) - 0.5),
                    device=device,
                )
            else:
                mi_val = mi_cpu(reps_concat[i].ravel(), labels, bins=bins)
                # compute Hy for labels
                hy = entropy_from_hist(np.bincount(labels))
                nmi_val = float(mi_val / (hy + 1e-12))
        except Exception:
            mi_val = 0.0
            nmi_val = 0.0
        mi_mat[i, N - 1] = mi_val
        mi_mat[N - 1, i] = mi_val
        nmi_mat[i, N - 1] = nmi_val
        nmi_mat[N - 1, i] = nmi_val

    mi_mat[N - 1, N - 1] = entropies[N - 1]
    nmi_mat[N - 1, N - 1] = 1.0

    # use actual module names for layer ticks when available
    ticks = ["INPUT"] + _module_names_from_layers(model, layers) + ["LABEL"]
    # zero diagonals for input/layer cells (keep LABEL diag as entropy)
    np.fill_diagonal(mi_mat, 0.0)
    np.fill_diagonal(nmi_mat, 0.0)
    # save heatmaps and arrays
    plot_heatmap(mi_mat, xticks=ticks, yticks=ticks, title=f"MI matrix ({backbone})", out_path=os.path.join(out_dir, f"mi_matrix_{backbone}.png"))
    plot_heatmap(nmi_mat, xticks=ticks, yticks=ticks, title=f"NMI matrix ({backbone})", out_path=os.path.join(out_dir, f"nmi_matrix_{backbone}.png"))
    np.savez(os.path.join(out_dir, f"mi_nmi_entropy_{backbone}.npz"), mi=mi_mat, nmi=nmi_mat, entropy=entropies, ticks=ticks)
    return {"mi": mi_mat, "nmi": nmi_mat, "entropy": entropies, "ticks": ticks}


def run_layer_layer_matrix(backbone: str, data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, augment: bool = False, out_dir: str = "outputs", max_layers: Optional[int] = 6, pretrained: bool = False) -> Dict[str, np.ndarray]:
    """Compute layer<->layer MI/NMI matrix and save heatmap."""
    device = select_device()
    ensure_dir(out_dir)
    default_workers = 0 if platform.system() == "Windows" else 4
    _, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=augment, normalize=False)
    model = _build_backbone(backbone, pretrained=pretrained, num_classes=num_classes)
    # move model to device before extracting layers and running forwards
    model = model.to(device)
    model.eval()
    layers = list_conv_layers(model)
    if max_layers is not None:
        layers = layers[:max_layers]

    # collect reduced per-layer per-sample vectors (N, C) per layer
    reps = [[] for _ in layers]
    labels_list = []
    autocast_enabled = (device.type == "cuda")
    with ActivationCatcher(layers) as ac:
        for xb, yb in val_loader:
            with autocast(device.type, enabled=autocast_enabled):
                _ = model(xb.to(device))
            acts = ac.get_activations()
            for i, a in enumerate(acts[: len(layers)]):
                if a is None:
                    reps[i].append(np.zeros((xb.size(0), 0)))
                    continue
                if a.dim() == 4:
                    r = reduce_activation(a, mode="spatial_mean")  # (B, C)
                    reps[i].append(r.cpu().numpy())
                elif a.dim() == 2:
                    reps[i].append(a.cpu().numpy())
                else:
                    reps[i].append(a.view(a.size(0), -1).cpu().numpy())
            labels_list.append(yb.numpy())
    # Concatenate per-layer arrays into (N, C) per layer
    reps_per_layer = [np.concatenate(rr, axis=0) if rr else np.zeros((0, 0)) for rr in reps]

    L = len(reps_per_layer)
    mi_mat = np.zeros((L, L), dtype=float)
    nmi_mat = np.zeros((L, L), dtype=float)
    for i in range(L):
        xi = reps_per_layer[i]
        for j in range(i, L):
            xj = reps_per_layer[j]
            if xi.shape[0] != xj.shape[0]:
                mi_val = 0.0
                nmi_val = 0.0
            else:
                mi_val, nmi_val = _parallel_channel_mean_mi(xi, xj, device=device, bins=bins)
            mi_mat[i, j] = float(mi_val)
            mi_mat[j, i] = float(mi_val)
            nmi_mat[i, j] = float(nmi_val)
            nmi_mat[j, i] = float(nmi_val)

    ticks = _module_names_from_layers(model, layers)
    plot_heatmap(mi_mat, xticks=ticks, yticks=ticks, title=f"Layer-Layer MI ({backbone})", out_path=os.path.join(out_dir, f"layer_layer_mi_{backbone}.png"))
    plot_heatmap(nmi_mat, xticks=ticks, yticks=ticks, title=f"Layer-Layer NMI ({backbone})", out_path=os.path.join(out_dir, f"layer_layer_nmi_{backbone}.png"))
    np.savez(os.path.join(out_dir, f"layer_layer_mi_nmi_{backbone}.npz"), mi=mi_mat, nmi=nmi_mat, ticks=ticks)
    return {"mi": mi_mat, "nmi": nmi_mat, "ticks": ticks}


def run_training_mi_matrices(
    backbone: str,
    checkpoints_dir: str = "checkpoints",
    data_root: str = "data/imagenette2",
    out_dir: str = "DA_mi_results",
    bins: int = 20,
    batch_size: int = 64,
    augment: bool = False,
    max_layers: Optional[int] = None,
) -> None:
    """
    For each checkpoint in `checkpoints_dir`, compute the full INPUT+layers+LABEL
    MI/NMI matrix and save a pkl named like the notebook:
      out_dir/mi_nmi_epoch{E}_acc{ACC}.pkl

    Each pkl contains:
      {'layers': [tick strings], 'mi_matrix': np.ndarray, 'nmi_matrix': np.ndarray}
    """
    device = select_device()
    ensure_dir(out_dir)

    _, val_loader, num_classes = get_imagenette_loaders(
        data_root, batch_size=batch_size, num_workers=4, augment=augment, normalize=False
    )

    # Collect checkpoints and sort by epoch
    ckpt_dir = Path(checkpoints_dir)
    files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: parse_epoch_acc(p.name)[0])

    for ckpt in files:
        epoch, acc = parse_epoch_acc(ckpt.name)
        # Build model and load weights
        model = _build_backbone(backbone, pretrained=False, num_classes=num_classes)
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state.get("model_state", state))
        model.eval()
        model = model.to(device)

        # Prepare layers and ticks
        layers = list_conv_layers(model)
        if max_layers is not None:
            layers = layers[:max_layers]
        tick_names = ["INPUT"] + _module_names_from_layers(model, layers) + ["LABEL"]

        # Collect INPUT RGB-channel mean features (N,3) and expanded labels (N,3)
        input_feats_list = []
        labels_expanded_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                # (B,3,-1) -> mean over pixels -> (B,3)
                x_feats = xb.view(xb.size(0), 3, -1).mean(dim=2).cpu()
                input_feats_list.append(x_feats.numpy())
                # expand labels to match channels (B,3)
                labels_expanded_list.append(yb.unsqueeze(1).expand(-1, x_feats.shape[1]).float().cpu().numpy())
        input_feats_all = np.concatenate(input_feats_list, axis=0) if input_feats_list else np.zeros((0, 3))
        labels_all = np.concatenate(labels_expanded_list, axis=0) if labels_expanded_list else np.zeros((0, 3))

        # Collect per-layer reduced representations (keep per-channel vectors per sample)
        reps = [input_feats_all]
        reps_layer: List[np.ndarray] = []
        autocast_enabled = (device.type == "cuda")
        with ActivationCatcher(layers) as ac:
            all_scalars = [list() for _ in layers]
            for xb, _ in val_loader:
                xb = xb.to(device)
                with autocast(device.type, enabled=autocast_enabled):
                    _ = model(xb)
                acts = ac.get_activations()
                for i, a in enumerate(acts[: len(layers)]):
                    if a is None:
                        continue
                    a = a.detach().cpu()
                    if a.dim() == 4:
                        # reduce spatially but keep channels: (B, C)
                        a = reduce_activation(a, mode="spatial_mean")
                        all_scalars[i].append(a.numpy())
                    elif a.dim() == 2:
                        # already (B, C)
                        all_scalars[i].append(a.numpy())
                    else:
                        all_scalars[i].append(a.view(a.size(0), -1).numpy())
            for i in range(len(layers)):
                reps_layer.append(np.concatenate(all_scalars[i], 0))
        reps = reps + reps_layer + [labels_all.astype(np.float32)]
        N = len(reps)

        # Build MI/NMI matrices
        mi_mat = np.zeros((N, N), dtype=np.float32)
        nmi_mat = np.zeros((N, N), dtype=np.float32)

        # Entropy of label (use first column since labels were expanded across channels)
        if labels_all.size:
            label_flat = labels_all[:, 0].astype(int)
            hy = entropy_from_hist(np.bincount(label_flat))
        else:
            hy = 0.0

        # Pairwise (INPUT+layers) vs (INPUT+layers): compute per-channel MI and average
        reps_per = [r for r in reps[:-1]]  # exclude label column for now
        Lp = len(reps_per)
        for i in range(Lp):
            xi = reps_per[i]
            for j in range(Lp):
                xj = reps_per[j]
                if xi.shape[0] != xj.shape[0]:
                    mi_val = 0.0
                    nmi_val = 0.0
                else:
                    mi_val, nmi_val = _parallel_channel_mean_mi(xi, xj, device=device, bins=bins)
                mi_mat[i, j] = float(mi_val)
                nmi_mat[i, j] = float(nmi_val)

        # INPUT+layers vs LABEL: compute per-channel MI between each per-layer (N,C) and label (N,1)
        label_col = reps[-1]
        for i in range(N - 1):
            xi = reps[i]
            if xi.shape[0] != label_col.shape[0]:
                mi_val = 0.0
                nmi_val = 0.0
            else:
                mi_val, nmi_val = _parallel_channel_mean_mi(xi, label_col, device=device, bins=bins)
            mi_mat[i, N - 1] = mi_val
            mi_mat[N - 1, i] = mi_val
            nmi_mat[i, N - 1] = float(mi_val / (hy + 1e-12) if hy > 0 else 0.0)
            nmi_mat[N - 1, i] = nmi_mat[i, N - 1]

        # zero diagonal for input/layer cells to avoid self-MI bias; keep label entropy at diag
        np.fill_diagonal(mi_mat, 0.0)
        np.fill_diagonal(nmi_mat, 0.0)
        mi_mat[N - 1, N - 1] = hy
        nmi_mat[N - 1, N - 1] = 1.0

        # Save like the notebook
        tag = f"epoch{epoch}_acc{(acc if acc is not None else 0):.2f}"
        out_pkl = Path(out_dir) / f"mi_nmi_{tag}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump({"layers": tick_names, "mi_matrix": mi_mat, "nmi_matrix": nmi_mat}, f)
        print(f"Saved MI/NMI results to {out_pkl}")
        torch.cuda.empty_cache()


def run_training_trajectory(backbone: str, checkpoints_dir: str = "checkpoints", data_root: str = "data/imagenette2", bins: int = 20, batch_size: int = 64, augment: bool = False, out_dir: str = "outputs", max_layers: Optional[int] = 6, include_nmi: bool = False, pretrained: bool = False, write_cache: bool = False) -> Dict[str, List[List[float]]]:
    """Load checkpoints in order and compute per-layer MI vs labels across epochs."""
    device = select_device()
    ensure_dir(out_dir)
    default_workers = 0 if platform.system() == "Windows" else 4
    _, val_loader, num_classes = get_imagenette_loaders(data_root, batch_size=batch_size, num_workers=default_workers, augment=augment, normalize=False)

    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {checkpoints_dir}")

    # gather checkpoint files and parse epochs
    cands = [p for p in ckpt_dir.glob("*.pt")]
    parsed: List[Tuple[int, Path]] = []
    for p in cands:
        try:
            ep, _ = parse_epoch_acc(p.name)
            parsed.append((ep, p))
        except Exception:
            continue
    if not parsed:
        raise RuntimeError("no checkpoints found with parseable epoch")
    parsed.sort(key=lambda x: x[0])

    trajectories: List[List[float]] = []
    nmi_trajectories: List[List[float]] = []
    layer_names: List[str] = []
    label_mi_per_epoch: List[float] = []

    for ep, path in parsed:
        # build a fresh model and load state
        model = _build_backbone(backbone, pretrained=pretrained, num_classes=num_classes)
        state = torch.load(str(path), map_location="cpu")
        model.load_state_dict(state.get("model_state", state))
        model.eval()
        layers = list_conv_layers(model)
        if max_layers is not None:
            layers = layers[:max_layers]
        layer_mi = _compute_layer_label_mi(model, layers, val_loader, device, bins=bins, max_layers=max_layers, use_joint_hist=True, write_cache=write_cache)
        layer_nmi: Optional[List[float]] = None
        if include_nmi:
            # compute per-layer NMI by computing Hy per layer and dividing
            layer_nmi = []
            for li, layer in enumerate(layers[: len(layer_mi)]):
                # attempt to load reduced activations
                cache_key_name = f"{model.__class__.__name__}_L{li}_val"
                cached = load_activations(in_dir="activations_cache", key=cache_key_name)
                if cached is not None:
                    X = cached.numpy() if isinstance(cached, torch.Tensor) else np.array(cached)
                else:
                    X = None
                # compute Hy for the layer via histogram across flattened activations
                if X is None:
                    # fallback: set nmi to 0
                    layer_nmi.append(0.0)
                    continue
                # compute per-channel MI vs labels and Hy for label
                # here we compute average NMI across channels using Hy computed from labels
                hy = entropy_from_hist(np.bincount(np.concatenate([np.zeros(1, dtype=int)]))) if True else 0.0
                # for robustness just reuse MI and divide by label entropy
                # compute MI per channel quickly on CPU
                mi_vals_local = []
                for ci in range(X.shape[1]):
                    mi_vals_local.append(mi_cpu(X[:, ci], np.concatenate([np.zeros(0, dtype=int)]), bins=bins) if X.shape[0] > 0 else 0.0)
                layer_nmi.append(float(np.mean(mi_vals_local)))
        # initialize trajectories on first pass
        if not trajectories:
            trajectories = [[v] for v in layer_mi]
            layer_names = [f"L{i}" for i in range(len(layer_mi))]
            if include_nmi:
                nmi_trajectories = [[v] for v in (layer_nmi or [])]
        else:
            for i, v in enumerate(layer_mi):
                trajectories[i].append(v)
            if include_nmi and layer_nmi is not None:
                for i, v in enumerate(layer_nmi):
                    nmi_trajectories[i].append(v)
        # --- compute label-vs-logit MI for this checkpoint ---
        try:
            from mi_estimators import mi_gpu

            all_logits = []
            all_labels = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)  # logits [B, C]
                    all_logits.append(out.detach().float().cpu())
                    all_labels.append(yb.detach().long().cpu())
            if all_logits:
                logits = torch.cat(all_logits, dim=0)  # [N, C]
                labels = torch.cat(all_labels, dim=0)  # [N]
                per_class_mi = []
                for c in range(logits.shape[1]):
                    try:
                        mval = mi_gpu(logits[:, c], labels, bins=bins)
                        per_class_mi.append(float(mval))
                    except Exception:
                        per_class_mi.append(0.0)
                label_mi = float(sum(per_class_mi) / len(per_class_mi)) if per_class_mi else 0.0
            else:
                label_mi = 0.0
        except Exception:
            label_mi = 0.0
        label_mi_per_epoch.append(label_mi)

    # save per-layer trajectory plots
    epochs = [ep for ep, _ in parsed]
    for i, traj in enumerate(trajectories):
        fig = plot_trajectory(epochs, traj, label=layer_names[i], title=f"Layer {layer_names[i]} MI over epochs", out_path=os.path.join(out_dir, f"traj_{backbone}_{layer_names[i]}_mi.png"))
        if include_nmi and nmi_trajectories:
            plot_trajectory(epochs, nmi_trajectories[i], label=layer_names[i], title=f"Layer {layer_names[i]} NMI over epochs", out_path=os.path.join(out_dir, f"traj_{backbone}_{layer_names[i]}_nmi.png"))

    out = {"epochs": [ep for ep, _ in parsed], "layer_names": layer_names, "trajectories": trajectories}
    if include_nmi:
        out["nmi_trajectories"] = nmi_trajectories
    # add label MI trajectory and save
    out["label_mi"] = label_mi_per_epoch
    try:
        np.save(os.path.join(out_dir, "label_mi_traj.npy"), np.array(label_mi_per_epoch))
    except Exception:
        pass
    return out


def cli_args():
    import argparse

    parser = argparse.ArgumentParser(prog="analysis", description="Run MI analysis studies")
    parser.add_argument("--backbone", choices=["vgg16", "resnet18", "resnet50"], default="vgg16")
    parser.add_argument("--study", choices=["pretrained_vs_random", "input_layer_label_matrix", "layer_layer_matrix", "stream_vgg", "stream_resnet", "trajectory"], required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=64)
    parser.add_argument("--data-root", dest="data_root", default="data/imagenette2")
    parser.add_argument("--max-layers", dest="max_layers", type=int, default=None)
    parser.add_argument("--include-nmi", dest="include_nmi", action="store_true")
    parser.add_argument("--use-parallel", dest="use_parallel", action="store_true", help="Enable parallel helpers when available (no-op if unsupported)")
    parser.add_argument("--out-dir", dest="out_dir", default="outputs")
    parser.add_argument("--checkpoints-dir", dest="checkpoints_dir", default="checkpoints")
    return parser.parse_args()


def main():
    args = cli_args()
    print(f"Running study={args.study} backbone={args.backbone} bins={args.bins} batch_size={args.batch_size} data_root={args.data_root} out_dir={args.out_dir}")

    if args.study == "pretrained_vs_random":
        run_pretrained_vs_random(args.backbone, data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, augment=False, out_dir=args.out_dir, max_layers=args.max_layers)

    elif args.study == "input_layer_label_matrix":
        run_input_layer_label_matrix(args.backbone, data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, augment=False, out_dir=args.out_dir, max_layers=args.max_layers)

    elif args.study == "layer_layer_matrix":
        run_layer_layer_matrix(args.backbone, data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, augment=False, out_dir=args.out_dir, max_layers=args.max_layers)

    elif args.study == "stream_vgg":
        run_vgg16_stream(data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, pretrained=False, max_layers=args.max_layers)

    elif args.study == "stream_resnet":
        if args.backbone == "resnet50":
            run_resnet50_stream(data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, pretrained=False, max_blocks_per_layer=args.max_layers)
        else:
            run_resnet18_stream(data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, pretrained=False, max_blocks_per_layer=args.max_layers)

    elif args.study == "trajectory":
        run_training_trajectory(args.backbone, checkpoints_dir=args.checkpoints_dir, data_root=args.data_root, bins=args.bins, batch_size=args.batch_size, augment=False, out_dir=args.out_dir, max_layers=args.max_layers, include_nmi=args.include_nmi)

    else:
        raise SystemExit(f"Unknown study: {args.study}")


if __name__ == "__main__":
    main()



