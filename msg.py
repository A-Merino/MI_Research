import infomeasure as im
import numpy as np
from time import time as tt
from typing import Optional, Tuple, List, Union


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



rng = np.random.default_rng(692475)

rho = 0.5
data = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=1000)
x, y = data[:, 0], data[:, 1]




from scipy.special import digamma
from scipy.spatial import KDTree

def torch_ksg(x, y, k=4, noise_level=0.0, minkowski_p=np.inf, normalize=False):
    N = len(x)
    
    data = np.column_stack((x, y))
    tree = KDTree(data)  # default leafsize=10 
    distances = tree.query(data, k=k, p=minkowski_p)[0]
    print(distances)


    # number of datapoints


    mi = digamma(k) + digamma(N) - (1 / N)

    return mi    


print(mi_cpu(x, y), # regular mi
    mi2d_cpu(x, y),  # 2d 
    torch_ksg(x, y),  # ksg
    -0.5 * np.log(1 - rho**2))  # analytical value

# s = tt()
# mi = mi_cpu(x, y)
# e = tt()
# print(f'Seconds to calc: {e-s:0.6f}')
# print(f'MI value: {mi:0.7f}')

# s = tt()
# mi = mi2d_cpu(x, y)
# e = tt()
# print(f'Seconds to calc: {e-s:0.6f}')
# print(f'MI value: {mi[0]:0.7f}')



# s = tt()
# mi = im.mutual_information(x, y, approach="discrete")
# e = tt()
# print(f'Seconds to calc: {e-s:0.6f}')
# print(f'MI value: {mi:0.7f}')

# s = tt()
# mi = im.mutual_information(x, y, approach="metric")
# e = tt()
# print(f'Seconds to calc: {e-s:0.6f}')
# print(f'MI value: {mi:0.7f}')

# # s = tt()
# # mi = im.mutual_information(x, y, approach="kernel", kernel="gaussian", bandwidth=0.7)
# # e = tt()
# # print(f'Seconds to calc: {e-s:0.6f}')
# # print(f'MI value: {mi:0.7f}')

# s = tt()
# mi = im.mutual_information(x, y, approach="kernel", kernel="box", bandwidth=0.7)
# e = tt()
# print(f'Seconds to calc: {e-s:0.6f}')
# print(f'MI value: {mi:0.7f}')
