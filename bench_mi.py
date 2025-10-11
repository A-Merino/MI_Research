# bench_mi.py
"""
MI timing benchmark.

Produces timing results for a single-feature vector of length N=1_000_000.
Uses the CPU binning MI (bins=20) implemented in `mi_estimators.mi_cpu` and
the 2D histogram GPU-friendly estimator `mi_estimators.mi2d_gpu` (bins=20).

Timing protocol:
- Warm-up runs (not timed)
- 13 iterations; average the last 10 (drop first 3)
- Wrap GPU calls with torch.cuda.synchronize() when CUDA is available

If no CUDA device is available, the GPU estimator is still invoked on CPU
so the script completes and reports timings.
"""

import numpy as np
import torch
from timeit import default_timer as timer
from mi_estimators import mi_cpu, mi2d_gpu


def smoke_label_noise_wrapper():
    """Quick smoke test for SymmetricLabelNoise wrapper.

    Wrap a small slice of the training ImageFolder dataset and print:
      - dataset length
      - number of flipped labels
      - histogram of noisy labels
    """
    from torchvision import datasets
    from datasets import get_imagenette_loaders, SymmetricLabelNoise
    import collections

    # Build a tiny loader to access the ImageFolder dataset (no workers)
    try:
        train_loader, _, num_classes = get_imagenette_loaders(data_root="data/imagenette2", batch_size=8, num_workers=0, augment=False, normalize=False)
    except Exception as e:
        print("Could not build imagenette loaders:", e)
        return

    base = train_loader.dataset
    # take a small slice of the dataset for quick testing
    n = min(128, len(base))
    # monkeypatch samples to a small slice by creating a shallow copy
    small_base = type(base)(base.root, transform=base.transform)
    small_base.samples = base.samples[:n]
    small_base.targets = [s[1] for s in small_base.samples]

    noised = SymmetricLabelNoise(small_base, noise_rate=0.3, num_classes=num_classes, seed=0)
    total = len(noised)
    mapping = getattr(noised, "mapping", None)
    if mapping is not None:
        num_flips = int((mapping[:, 0] != mapping[:, 1]).sum().item())
    else:
        # fallback: compare original targets
        orig = [s[1] for s in small_base.samples]
        noisy_targets = noised.targets
        num_flips = sum(1 for o, n in zip(orig, noisy_targets) if o != n)

    # histogram of noisy labels
    hist = collections.Counter(getattr(noised, "targets", [s[1] for s in noised.samples]))

    print(f"Dataset length: {total}")
    print(f"Number of flips: {num_flips}")
    print(f"Noisy label histogram: {dict(sorted(hist.items()))}")


def main():
    N = 1_000_000
    torch.manual_seed(0)
    np.random.seed(0)

    # single feature and integer labels 0..9 (cast to float per notebook)
    features = torch.randn(N)
    labels = torch.randint(0, 10, (N,)).float()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Warm-up
    _ = mi_cpu(features.numpy(), labels.numpy(), bins=20)
    if use_cuda:
        f_gpu = features.to(device)
        l_gpu = labels.to(device)
        torch.cuda.synchronize()
    # compute deterministic x_range for these synthetic features
    x_min = float(features.min().item())
    x_max = float(features.max().item())
    _ = mi2d_gpu(f_gpu, l_gpu, bins=(20, 10), x_range=(x_min, x_max), y_range=(-0.5, 9.5), device=device)

    # CPU timing
    cpu_times = []
    for i in range(13):
        t0 = timer()
        _ = mi_cpu(features.numpy(), labels.numpy(), bins=20)
        t1 = timer()
        if i > 2:
            cpu_times.append(t1 - t0)

    # GPU timing (or run mi2d_gpu on CPU if no CUDA)
    gpu_times = []
    if use_cuda:
        for i in range(13):
            torch.cuda.synchronize()
            t0 = timer()
            _ = mi2d_gpu(f_gpu, l_gpu, bins=(20, 10), x_range=(x_min, x_max), y_range=(-0.5, 9.5), device=device)
            torch.cuda.synchronize()
            t1 = timer()
            if i > 2:
                gpu_times.append(t1 - t0)
    else:
        for i in range(13):
            t0 = timer()
            _ = mi2d_gpu(features, labels, bins=(20, 10), x_range=(float(features.min().item()), float(features.max().item())), y_range=(-0.5, 9.5), device=torch.device("cpu"))
            t1 = timer()
            if i > 2:
                gpu_times.append(t1 - t0)

    # Final MI values
    mi_cpu_val = mi_cpu(features.numpy(), labels.numpy(), bins=20)
    if use_cuda:
        mi_gpu_val = mi2d_gpu(f_gpu, l_gpu, bins=(20, 10), x_range=(x_min, x_max), y_range=(-0.5, 9.5), device=device)[0]
    else:
        mi_gpu_val = mi2d_gpu(features, labels, bins=(20, 10), x_range=(float(features.min().item()), float(features.max().item())), y_range=(-0.5, 9.5), device=torch.device("cpu"))[0]

    print(f"CPU MI: {mi_cpu_val:.4f} | Avg Time: {np.mean(cpu_times):.4f} sec")
    print(f"GPU MI: {mi_gpu_val:.4f} | Avg Time: {np.mean(gpu_times):.4f} sec (device: {device.type})")


if __name__ == "__main__":
    main()
