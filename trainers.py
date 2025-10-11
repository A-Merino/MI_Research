"""
trainers.py
Lightweight training loop and checkpoint saving.

This module will provide a simple train/validate loop, optimizer setup, and
checkpoint utilities used by the experiments scripts.
"""

from typing import Any, Dict, List, Optional
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from utils import Timer, ensure_dir
from typing import Tuple


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(p))


def _evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return {"loss": avg_loss, "acc": acc}


def train_classifier(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    epochs: int = 50,
    lr: float = 1e-2,
    momentum: float = 0.9,
    step_size: int = 5,
    gamma: float = 0.1,
    early_patience: int = 7,
    out_dir: str = "checkpoints",
    save_acc_in_name: bool = True,
    model_tag: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train a classifier with SGD and cross-entropy, save best checkpoint.

    Returns a history dict containing lists: train_loss, val_loss, train_acc, val_acc.
    """
    device_obj = torch.device(device if isinstance(device, str) else str(device))
    model = model.to(device_obj)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    # StepLR to roughly match notebook's step-based LR schedule
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()

    ensure_dir(out_dir)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = Timer()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_metrics = _evaluate(model, val_loader, device_obj)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = t0.__enter__() if False else None
        # Log per-epoch stats with visible LR (percentage accuracy formatting)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_acc={train_acc*100:.2f}% | val_acc={val_acc*100:.2f}% | LR={cur_lr}")

        # Checkpointing — include model_tag in checkpoint filenames when provided
        tag_part = f"_{model_tag}" if model_tag else ""
        ckpt_name = f"epoch{epoch:03d}_acc{val_acc:.3f}{tag_part}.pt"
        ckpt_path = os.path.join(out_dir, ckpt_name)
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_acc, "tag": model_tag}
        save_checkpoint(ckpt_path, state)

        # Step the scheduler once per epoch
        try:
            scheduler.step()
        except Exception:
            pass

        # track best for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_patience is not None and epochs_no_improve >= early_patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch} val_acc={best_val_acc:.3f})")
            break

    # record tag in history for downstream bookkeeping
    history["tag"] = model_tag
    return history


