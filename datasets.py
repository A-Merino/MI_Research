"""datasets — Imagenette dataset loaders and notebook-matching transforms

Provides ``get_imagenette_loaders`` that returns (train_loader, val_loader, num_classes).
Validation transforms match the notebook: Resize((224,224)) + ToTensor().
"""

from pathlib import Path
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T


class SymmetricLabelNoise(torch.utils.data.Dataset):
    """Dataset wrapper that applies symmetric label noise to an ImageFolder-like dataset.

    Builds a new `samples` list from `base.samples` where labels are flipped with
    probability `noise_rate` to a uniformly random *different* class. Records a
    `mapping` tensor of shape (N,2) with columns (original_label, noisy_label).
    The wrapper preserves the wrapped dataset's `loader`, `transform`, and
    `target_transform`.
    """

    def __init__(self, base, noise_rate: float, num_classes: int, seed: int | None = None):
        self.base = base
        self.noise_rate = float(noise_rate)
        self.num_classes = int(num_classes)
        g = torch.Generator()
        g.manual_seed(0 if seed is None else int(seed))

        # ImageFolder exposes .samples as list[(path, label)]
        paths, labels = zip(*self.base.samples)
        labels = torch.tensor(labels, dtype=torch.long)
        n = labels.numel()

        flip_mask = torch.rand(n, generator=g) < self.noise_rate
        rand_classes = torch.randint(low=0, high=self.num_classes, size=(n,), generator=g)
        # ensure sampled class is different from original by shifting duplicates
        rand_classes = (rand_classes + (rand_classes == labels).to(torch.long)) % self.num_classes

        noisy = labels.clone()
        noisy[flip_mask] = rand_classes[flip_mask]

        self.mapping = torch.stack([labels, noisy], dim=1)
        self.samples = [(p, int(y)) for p, y in zip(paths, noisy.tolist())]
        self.targets = [s[1] for s in self.samples]
        # preserve class metadata from base when available
        self.classes = getattr(self.base, "classes", list(range(self.num_classes)))
        self.class_to_idx = getattr(self.base, "class_to_idx", {c: i for i, c in enumerate(self.classes)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.base.loader(path)
        transform = getattr(self.base, "transform", None)
        target_transform = getattr(self.base, "target_transform", None)
        img = transform(img) if transform else img
        target = target_transform(target) if target_transform else target
        return img, target


def get_imagenette_loaders(
    root: str = "data/imagenette2",
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = False,
    data_root: str = None,
    normalize: bool = False,
    label_noise: float = 0.0,
    num_classes: int = 10,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val DataLoaders for Imagenette matching the notebook transforms.

    Validation defaults to non-normalized transforms (Resize(224), ToTensor()). Pass
    ``normalize=True`` when you want ImageNet normalization for training/eval.

    The exact validation transform used when ``normalize=True`` is:

        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    Resize((224,224)) and ToTensor() are always applied; the Normalize step is
    appended only when ``normalize=True``.

    Returns (train_loader, val_loader, num_classes).
    """

    # support legacy callers passing `data_root=`
    if data_root is not None:
        root = Path(data_root)
    else:
        root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected 'train/' and 'val/' under {root!s}")

    # Build validation transform (Resize + ToTensor + optional ImageNet normalization)
    val_tf_list = [T.Resize((224, 224)), T.ToTensor()]
    if normalize:
        val_tf_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    val_tf = T.Compose(val_tf_list)

    # Build training transform lists and append normalization when requested
    if augment:
        train_tf_list = [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]
    else:
        train_tf_list = [T.Resize((224, 224)), T.ToTensor()]
    if normalize:
        train_tf_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    train_tf = T.Compose(train_tf_list)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    # Robust class count: use classes from train; ensure val classes consistent
    # actual number of classes in the dataset
    num_classes = len(train_ds.classes)
    if len(val_ds.classes) != num_classes:
        raise ValueError("Train/val class sets are inconsistent")

    pin_memory = torch.cuda.is_available()

    # If requested, wrap the training dataset with symmetric label noise.
    # Noise is applied only to the train dataset and labels are precomputed once.
    if label_noise and label_noise > 0.0:
        # Use the dataset's class count for noise unless the caller explicitly
        # provided a different `num_classes` param. We respect the caller's
        # `num_classes` argument as the number of classes to sample noise from.
        noise_num_classes = int(num_classes) if isinstance(num_classes, int) else int(num_classes)
        # However, prefer the actual dataset class count if it differs (safer).
        # If they differ, override the noise_num_classes to the dataset count.
        if noise_num_classes != num_classes:
            noise_num_classes = num_classes
        # Wrap
        train_ds = SymmetricLabelNoise(train_ds, noise_rate=label_noise, num_classes=num_classes, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    # Quick inline smoke test: iterate a single batch on CPU if data exists.
    import sys

    root = Path("data/imagenette2")
    if not (root / "train").exists():
        print(f"Data not found at {root.resolve()}, skipping smoke test")
        sys.exit(0)

    print("Creating loaders (smoke test) - CPU mode")
    train_loader, val_loader, nc = get_imagenette_loaders(
        data_root=str(root), batch_size=8, num_workers=0, augment=False, normalize=True
    )

    print(f"Found {nc} classes. Iterating one training batch...")
    images, labels = next(iter(train_loader))
    print("Batch images shape:", getattr(images, "shape", type(images)))
    print("Batch labels shape:", getattr(labels, "shape", type(labels)))


