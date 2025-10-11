"""hooks — Layer hook utilities and activation reducers

Utilities to attach forward hooks to model layers and reduce activations
for MI calculations (e.g., spatial mean, flatten, optional PCA).
"""

from typing import Any, Callable, List, Dict, Optional

import torch
import torch.nn as nn


def list_conv_layers(model: nn.Module) -> List[nn.Module]:
    """Return a list of convolutional layers (nn.Conv2d) in forward order.

    Works for common torchvision models such as VGG and ResNet by iterating
    the module tree in definition order and collecting Conv2d modules.
    """
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


class ActivationCatcher:
    """Context manager that registers forward hooks on provided layers and
    collects their outputs on CPU to limit GPU memory usage.

    Example:
        layers = list_conv_layers(model)[:5]
        with ActivationCatcher(layers) as ac:
            out = model(input)
        # ac.activations is a list of tensors on CPU in the same order as `layers`
    """

    def __init__(self, layers: List[nn.Module], detach: bool = True) -> None:
        self.layers = list(layers)
        self.detach = detach
        self.handles: List[Any] = []
        # store one activation per layer per forward (on CPU)
        self.activations: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        # seen flags prevent multiple writes in the same forward pass
        self._seen: List[bool] = [False] * len(self.layers)

    def _make_hook(self, idx: int) -> Callable:
        def hook(module: nn.Module, inp, outp):
            # Move activation to CPU and detach to avoid holding GPU memory.
            try:
                # only capture the first activation per forward for this module
                if self._seen[idx]:
                    return
                t = outp
                if isinstance(outp, (tuple, list)):
                    t = outp[0]
                if self.detach:
                    t = t.detach()
                # move to CPU and clone to avoid referencing GPU memory
                t_cpu = t.cpu().clone()
                self.activations[idx] = t_cpu
                self._seen[idx] = True
            except Exception:
                # If extraction fails, store None and continue
                self.activations[idx] = None

        return hook

    def __enter__(self) -> "ActivationCatcher":
        # register hooks
        for i, layer in enumerate(self.layers):
            handle = layer.register_forward_hook(self._make_hook(i))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # remove hooks
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def get_activations(self) -> List[Optional[torch.Tensor]]:
        """Return the collected activations (ordered as the input layers).

        Each entry is either a CPU tensor or None if the hook failed or the
        layer did not run.
        """
        out = list(self.activations)
        # reset stored activations and seen flags so subsequent forwards start fresh
        self.activations = [None] * len(self.layers)
        self._seen = [False] * len(self.layers)
        return out


def reduce_activation(act: torch.Tensor, mode: str = "spatial_mean", pca_k: Optional[int] = None) -> torch.Tensor:
    """Reduce an activation tensor with optional PCA projection.

    Args:
        act: activation tensor. Common shapes: (B,C,H,W) or (B,C) or (B, ...)
        mode: one of 'spatial_mean', 'flatten', 'global_avg'
        pca_k: if set, apply PCA to reduce the feature dimension to k

    Returns:
        Reduced tensor. If pca_k is None, returns the reduced activation. If
        pca_k is set, returns a tensor of shape (B, pca_k).
    
    Modes:
        - 'spatial_mean': for (B,C,H,W) returns (B,C) by averaging H and W.
        - 'flatten': reshapes each sample to a 1D vector (B, -1).
        - 'global_avg': returns (B,) by averaging over all non-batch dims.
    """
    if act is None:
        raise ValueError("Activation is None")

    t = act
    # Handle 4D activations commonly returned by conv layers
    if mode == "spatial_mean":
        if t.dim() == 4:
            t = t.mean(dim=(-2, -1))  # (B, C)
        elif t.dim() == 2:
            # already (B, C)
            t = t
        else:
            raise ValueError(f"spatial_mean expects input with 2 or 4 dims, got {t.dim()}")

    elif mode == "flatten":
        # collapse all non-batch dims
        if t.dim() == 1:
            t = t.unsqueeze(0)
        t = t.reshape(t.shape[0], -1)

    elif mode == "global_avg":
        # mean over all dims except batch -> (B,)
        dims = tuple(range(1, t.dim()))
        t = t.mean(dim=dims)

    else:
        raise ValueError(f"Unknown reduction mode: {mode}")

    # At this point t is either (B, C) or (B, D) or (B,)
    if pca_k is None:
        return t

    # PCA projection expects a 2D matrix (B, F). If t is (B,), expand to (B,1)
    if t.dim() == 1:
        t = t.unsqueeze(1)

    B, F = t.shape
    if pca_k <= 0 or pca_k > min(B, F):
        raise ValueError("pca_k must be >0 and <= min(batch_size, feature_dim)")

    # center data
    mean = t.mean(dim=0, keepdim=True)
    X = t - mean

    # compute SVD on centered data X (B x F): X = U S V^T
    # we want principal components in V (shape F x F), project rows onto first k V cols
    # Use torch.linalg.svd for stability; compute full_matrices=False
    # For efficiency, if F is large and B < F, consider SVD on X^T X, but keep generic here.
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # Vh is shape (min(B,F), F) with rows = principal directions; take first k rows
    V_k = Vh[:pca_k, :].to(dtype=t.dtype)
    # project: X (B,F) @ V_k.T -> (B, k)
    projected = X @ V_k.T
    return projected



