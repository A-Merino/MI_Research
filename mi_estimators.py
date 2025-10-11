"""mi_estimators — CPU/GPU mutual information (MI) estimators

Provides CPU (numpy) and GPU (torch) binned MI/NMI estimators used across
the analysis scripts. Key functions: ``mi_cpu``, ``mi2d_gpu``, ``mi_gpu``, and
channel-parallel helpers.
"""

from typing import Optional, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch


def entropy_from_hist(counts: np.ndarray) -> float:
    """Compute entropy (nats) from integer histogram counts.

    Ignores zero-count bins.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p_nonzero = p[p > 0]
    return -float(np.sum(p_nonzero * np.log(p_nonzero)))


def nmi_scalar(mi: float, hy: float) -> float:
    """Return a simple normalized mutual information scalar.

    Normalization used in the notebook: MI / H(Y). Small epsilon added to
    denominator for numerical stability.
    """
    return float(mi) / (float(hy) + 1e-12)


def mi_cpu(x: np.ndarray, y: np.ndarray, bins: int = 20, eps: float = 1e-12) -> float:
    """Estimate mutual information I(X;Y) by binning X and using class labels Y (CPU, numpy).

    Args:
        x: 1D numpy array of real values (shape: N,)
        y: 1D numpy array of integer labels (shape: N,)
        bins: number of bins for X
        eps: tiny value added to probabilities to stabilize logs

    Returns:
        MI in nats (float)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    n = x.shape[0]
    if n == 0:
        return 0.0

    labels, inverse = np.unique(y, return_inverse=True)
    k = labels.shape[0]

    # bin edges over full x range
    x_min, x_max = float(x.min()), float(x.max())
    if x_max == x_min:
        x_max = x_min + 1.0
    edges = np.linspace(x_min, x_max, bins + 1)

    joint_counts = np.zeros((k, bins), dtype=np.float64)
    for i in range(k):
        mask = inverse == i
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(x[mask], bins=edges)
        joint_counts[i, :] = counts

    total = joint_counts.sum()
    if total == 0:
        return 0.0

    p_xy = joint_counts / total
    # add eps before logs for numerical stability
    p_xy = p_xy + eps
    p_x = p_xy.sum(axis=0)
    p_y = p_xy.sum(axis=1)

    # avoid zeros
    p_x = p_x + eps
    p_y = p_y + eps

    # compute MI = sum p_xy * (log p_xy - log p_x - log p_y)
    log_term = np.log(p_xy) - np.log(p_x)[None, :] - np.log(p_y)[:, None]
    mi_val = float(np.sum(p_xy * log_term))
    return mi_val


def parallel_mi_cpu(features, labels, bins: int = 20, workers: int = 4):
    """Compute MI for each feature/column in parallel using threads.

    Args:
        features: 2D array-like (N, D) where each column is a feature/channel.
        labels: 1D array-like (N,) integer labels.
        bins: number of bins passed to `mi_cpu`.
        workers: number of worker threads to use.

    Returns:
        numpy.ndarray of shape (D,) with MI values per channel.

    Notes:
        This is an optional helper to match the notebook's parallel channel-wise MI
        experiment. It deliberately uses ThreadPoolExecutor because `mi_cpu` uses
        numpy and the work per-task is mostly Python/NumPy bound.
    """
    X = np.asarray(features)
    y = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError("features must be a 2D array (N, D)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have the same number of rows")

    N, D = X.shape
    mi_results = np.zeros((D,), dtype=np.float64)

    # fallback for invalid worker counts
    if workers is None or workers <= 0:
        workers = 4

    def _task(i):
        return i, mi_cpu(X[:, i], y, bins=bins)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, i) for i in range(D)]
        for fut in as_completed(futures):
            i, val = fut.result()
            mi_results[i] = val

    return mi_results





def mi2d_gpu(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: Union[int, Tuple[int, int]] = 20,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Compute 2D binned mutual information I(X;Y) on GPU using torch ops.

    Returns a tuple (mi, nmi) where nmi = mi / H(Y) computed on the same bins.

    This implementation bins x and y into `bins` bins each, builds a joint
    histogram via linearized bin indices and `torch.bincount` (GPU-friendly),
    then computes entropies and MI.
    """
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
    x = x.to(device).flatten().double()
    y = y.to(device).flatten()
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same length")
    n = x.numel()
    if n == 0:
        return 0.0, 0.0

    # normalize bins argument to (bins_x, bins_y)
    if isinstance(bins, int):
        bins_x = bins_y = bins
    else:
        bins_x, bins_y = bins

    # determine x range (fixed if provided)
    if x_range is not None:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    else:
        x_min = float(x.min().item())
        x_max = float(x.max().item())
        if x_max == x_min:
            x_max = x_min + 1.0

    # build bin edges like numpy.linspace(..., bins_x+1) and use torch.bucketize
    edges_x = torch.linspace(x_min, x_max, steps=(bins_x + 1), device=device, dtype=torch.double)
    # bucketize expects boundaries (interior edges) to produce indices 0..bins_x-1
    boundaries_x = edges_x[1:-1].to(dtype=torch.double)
    ix = torch.bucketize(x, boundaries_x).to(torch.long)

    # If y is integer/categorical, follow CPU behaviour: rows=classes, cols=bins
    if not torch.is_floating_point(y):
        labels, inverse = torch.unique(y, sorted=True, return_inverse=True)
        k = labels.numel()
        if k == 0:
            return 0.0, 0.0
        # if caller provided y_range or bins_y equals known number of classes, respect fixed bins
        # use bins_y as the number of label bins
        # linear index: class_index * bins_x + x_bin
        linear_idx = inverse.to(dtype=torch.long) * bins_x + ix
        counts = torch.bincount(linear_idx, minlength=(k * bins_x)).to(dtype=torch.double, device=device)
        joint = counts.reshape(k, bins_x)
        total = joint.sum()
        if total <= 0:
            return 0.0, 0.0

        pxy = joint / total
        pxy = pxy + eps
        p_x = pxy.sum(dim=0)
        p_y = pxy.sum(dim=1)

        h_x = -torch.sum(p_x * torch.log(p_x))
        h_y = -torch.sum(p_y * torch.log(p_y))
        h_xy = -torch.sum(pxy * torch.log(pxy))

        mi = float((h_x + h_y - h_xy).cpu().item())
        h_y_f = float(h_y.cpu().item())
        nmi = float(mi / (h_y_f + 1e-12)) if h_y_f > 0 else 0.0
        return mi, nmi

    # fallback: continuous y -> 2D bin both x and y into (bins_x, bins_y) bins
    y = y.to(dtype=torch.double)
    if y_range is not None:
        y_min, y_max = float(y_range[0]), float(y_range[1])
    else:
        y_min = float(y.min().item())
        y_max = float(y.max().item())
        if y_max == y_min:
            y_max = y_min + 1.0
    edges_y = torch.linspace(y_min, y_max, steps=(bins_y + 1), device=device, dtype=torch.double)
    boundaries_y = edges_y[1:-1].to(dtype=torch.double)
    iy = torch.bucketize(y, boundaries_y).to(torch.long)

    linear_idx = ix * bins_y + iy
    counts = torch.bincount(linear_idx, minlength=(bins_x * bins_y)).to(dtype=torch.double, device=device)
    joint = counts.reshape(bins_x, bins_y)
    total = joint.sum()
    if total <= 0:
        return 0.0, 0.0

    pxy = joint / total
    pxy = pxy + eps
    # px = marginal over rows (X), py = marginal over columns (Y)
    px = pxy.sum(dim=1)
    py = pxy.sum(dim=0)

    hx = -torch.sum(px * torch.log(px))
    hy = -torch.sum(py * torch.log(py))
    hxy = -torch.sum(pxy * torch.log(pxy))

    mi = float((hx + hy - hxy).cpu().item())
    hy_f = float(hy.cpu().item())
    nmi = float(mi / (hy_f + 1e-12)) if hy_f > 0 else 0.0
    return mi, nmi


def mi2d_cpu(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, Tuple[int, int]] = 20,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Compute 2D binned mutual information I(X;Y) on CPU using numpy.

    Returns (mi, nmi) where nmi = mi / H(Y). Behaves analogously to
    ``mi2d_gpu``: if `y` is integer/categorical, we treat rows as classes and
    compute joint counts with shape (k, bins), otherwise we bin both x and y
    into `bins` bins to form a (bins, bins) joint histogram.
    """
    x = np.asarray(x).ravel().astype(np.float64)
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    n = x.shape[0]
    if n == 0:
        return 0.0, 0.0

    # normalize bins argument
    if isinstance(bins, int):
        bins_x = bins_y = bins
    else:
        bins_x, bins_y = bins

    # categorical Y path (mirror mi_cpu behaviour)
    if not np.issubdtype(y.dtype, np.floating):
        labels, inverse = np.unique(y, return_inverse=True)
        k = labels.size
        if k == 0:
            return 0.0, 0.0

        if x_range is not None:
            x_min, x_max = float(x_range[0]), float(x_range[1])
        else:
            x_min, x_max = float(x.min()), float(x.max())
            if x_max == x_min:
                x_max = x_min + 1.0
        edges = np.linspace(x_min, x_max, bins_x + 1)

        joint_counts = np.zeros((k, bins_x), dtype=np.float64)
        for i in range(k):
            mask = inverse == i
            if mask.sum() == 0:
                continue
            counts, _ = np.histogram(x[mask], bins=edges)
            joint_counts[i, :] = counts

        total = joint_counts.sum()
        if total == 0:
            return 0.0, 0.0

        pxy = joint_counts / total
        pxy = pxy + eps
        p_x = pxy.sum(axis=0)
        p_y = pxy.sum(axis=1)

        h_x = -np.sum(p_x * np.log(p_x + eps))
        h_y = -np.sum(p_y * np.log(p_y + eps))
        h_xy = -np.sum(pxy * np.log(pxy + eps))

        mi = float(h_x + h_y - h_xy)
        nmi = float(mi / (h_y + 1e-12)) if h_y > 0 else 0.0
        return mi, nmi

    # continuous y path: bin both x and y into (bins_x, bins_y) bins
    y = y.astype(np.float64)
    if x_range is not None:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    else:
        x_min, x_max = float(x.min()), float(x.max())
        if x_max == x_min:
            x_max = x_min + 1.0
    if y_range is not None:
        y_min, y_max = float(y_range[0]), float(y_range[1])
    else:
        y_min, y_max = float(y.min()), float(y.max())
        if y_max == y_min:
            y_max = y_min + 1.0

    edges_x = np.linspace(x_min, x_max, bins_x + 1)
    edges_y = np.linspace(y_min, y_max, bins_y + 1)
    # digitize to 0..bins-1
    ix = np.digitize(x, edges_x[1:-1])
    iy = np.digitize(y, edges_y[1:-1])

    linear = ix * bins_y + iy
    counts = np.bincount(linear, minlength=bins_x * bins_y).astype(np.float64)
    joint = counts.reshape((bins_x, bins_y))
    total = joint.sum()
    if total == 0:
        return 0.0, 0.0

    pxy = joint / total
    pxy = pxy + eps
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    hx = -np.sum(px * np.log(px + eps))
    hy = -np.sum(py * np.log(py + eps))
    hxy = -np.sum(pxy * np.log(pxy + eps))

    mi = float(hx + hy - hxy)
    nmi = float(mi / (hy + 1e-12)) if hy > 0 else 0.0
    return mi, nmi


def mi_gpu(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: Union[int, Tuple[int, int]] = 20,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Thin wrapper returning only MI using `mi2d_gpu`.

    Accepts per-axis bins and optional x_range/y_range and forwards to mi2d_gpu.
    """
    mi, _ = mi2d_gpu(x, y, bins=bins, x_range=x_range, y_range=y_range, device=device)
    return mi


def mi_pairwise_channels_parallel(x: torch.Tensor, y: torch.Tensor, bins: int = 20, device0: str = 'cuda:0', device1: str = 'cuda:1') -> Tuple[float, float]:
    """Split channel-wise MI computation across two devices in parallel.

    Args:
        x: Tensor of shape (N, C) or (N,) (if 1D will be treated as (N,1)).
        y: Labels tensor shape (N,) integer.
        bins: number of bins for mi2d_gpu.
        device0/device1: device strings for the two workers.

    Returns:
        (mean_mi, mean_nmi) across all channels.
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if x.dim() != 2:
        raise ValueError("x must be 2D (N, C) or 1D (N,)")
    N, C = x.shape
    if y.numel() != N:
        raise ValueError("x and y must have matching first-dimension length")

    # split channels
    mid = C // 2
    slices: List[torch.Tensor] = [x[:, :mid], x[:, mid:]]

    def _worker(dev_str: str, x_part: torch.Tensor):
        dev = torch.device(dev_str)
        mi_list: List[float] = []
        nmi_list: List[float] = []
        # run per-channel MI on this device
        with torch.no_grad():
            # determine k from y if integer
            try:
                y_cpu = y.cpu().numpy()
                if not np.issubdtype(y_cpu.dtype, np.floating):
                    k = int(y_cpu.max()) + 1 if y_cpu.size else 0
                else:
                    k = None
            except Exception:
                k = None
            for ci in range(x_part.shape[1]):
                xi = x_part[:, ci].to(dev)
                yi = y.to(dev)
                if k is not None:
                    mi_val, nmi_val = mi2d_gpu(xi, yi, bins=(bins, k), x_range=None, y_range=(-0.5, float(k) - 0.5), device=dev)
                else:
                    mi_val, nmi_val = mi2d_gpu(xi, yi, bins=bins, device=dev)
                mi_list.append(mi_val)
                nmi_list.append(nmi_val)
        return mi_list, nmi_list

    results_mi: List[float] = []
    results_nmi: List[float] = []
    # run in two threads to overlap GPU work
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = []
        futures.append(ex.submit(_worker, device0, slices[0]))
        futures.append(ex.submit(_worker, device1, slices[1]))
        for fut in as_completed(futures):
            mi_list, nmi_list = fut.result()
            results_mi.extend(mi_list)
            results_nmi.extend(nmi_list)

    if len(results_mi) == 0:
        return 0.0, 0.0
    mean_mi = float(np.mean(np.array(results_mi, dtype=np.float64)))
    mean_nmi = float(np.mean(np.array(results_nmi, dtype=np.float64)))
    return mean_mi, mean_nmi





if __name__ == "__main__":
    # Quick numerical sanity checks
    import numpy as _np
    import torch as _torch

    # Create correlated data: two Gaussians for two classes
    rng = _np.random.RandomState(0)
    N = 2000
    x = _np.concatenate([rng.normal(loc=-1.0, scale=0.5, size=N // 2), rng.normal(loc=1.0, scale=0.5, size=N // 2)])
    y = _np.concatenate([_np.zeros(N // 2, dtype=int), _np.ones(N // 2, dtype=int)])

    mi_c = mi_cpu(x, y, bins=20)
    print("mi_cpu:", mi_c)

    x_t = _torch.from_numpy(x.astype(_np.float32))
    y_t = _torch.from_numpy(y.astype(_np.int64))
    mi_g = mi_gpu(x_t, y_t, bins=20, device=_torch.device('cpu'))
    print("mi_gpu:", mi_g)

    assert mi_c > 0
    assert abs(mi_c - mi_g) < 1e-6 or abs(mi_c - mi_g) / max(1.0, abs(mi_c)) < 1e-3


