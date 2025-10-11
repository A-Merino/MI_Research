"""plots — Matplotlib plotting helpers for MI experiments

Lightweight functions to create and save figures headlessly (Agg backend):
`plot_mi_per_layer`, `plot_heatmap`, and `plot_trajectory`.
"""

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_mi_per_layer(layer_names: List[str], mi_pre: List[float], mi_rand: Optional[List[float]] = None, backbone_name: str = "", out_path: Optional[str] = None, title: Optional[str] = None, label_pre: str = "pretrained", label_rand: str = "random") -> Figure:
    """Line plot of MI per layer for one or two series (pretrained vs random).

    If `mi_rand` is None, a single series is plotted and x-axis tick labels use
    the provided `layer_names`. If both series are provided, both are plotted.

    Args:
        layer_names: list of layer name strings (x-axis)
        mi_pre: MI values for the first series (same length as layer_names)
        mi_rand: optional MI values for the second series
        backbone_name: short backbone label used in default title/legend
        out_path: optional path to save the PNG
        title: optional explicit title to override the default
        label_pre: label for the first series
        label_rand: label for the second series

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(max(6, len(layer_names) * 0.6), 4))
    x = np.arange(len(layer_names))
    # Always use the exact provided legend labels for the series.
    pre_label = f"{label_pre} {backbone_name}" if backbone_name else label_pre
    ax.plot(x, mi_pre, marker="o", linestyle="-", label=pre_label)
    if mi_rand is not None:
        rand_label = f"{label_rand} {backbone_name}" if backbone_name else label_rand
        ax.plot(x, mi_rand, marker="s", linestyle="--", label=rand_label)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel("Mutual Information (nats)")
    # default title if not provided. If both series are present, make title
    # explicitly mention pretrained vs random for clarity; otherwise use the
    # provided or default single-series title.
    if title is None:
        if mi_rand is None:
            ax.set_title(f"Mutual Information (Feature vs Label) per Conv Layer ({backbone_name})" if backbone_name else "Mutual Information (Feature vs Label) per Conv Layer")
        else:
            # make it explicit this is a comparison plot
            ax.set_title(f"Mutual Information per Conv Layer: {label_pre} vs {label_rand} ({backbone_name})" if backbone_name else f"Mutual Information per Conv Layer: {label_pre} vs {label_rand}")
    else:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    # Only show legend entries that were actually plotted. If mi_rand is None,
    # only the pretrained series will be labeled with label_pre.
    ax.legend()
    # x-axis label per-backbone exact wording
    if backbone_name.lower().startswith("vgg"):
        ax.set_xlabel("VGG Conv Layer")
    elif backbone_name.lower().startswith("resnet18"):
        ax.set_xlabel("ResNet18 Layer")
    elif backbone_name.lower().startswith("resnet50"):
        ax.set_xlabel("ResNet50 Layer")
    else:
        ax.set_xlabel("Layer")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_heatmap(matrix: np.ndarray, xticks: List[str], yticks: List[str], title: str = "", out_path: Optional[str] = None) -> Figure:
    """Plot a heatmap for `matrix` with provided axis tick labels.

    Args:
        matrix: 2D numpy array
        xticks: labels for columns
        yticks: labels for rows
        title: plot title
        out_path: optional path to save the figure

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_yticklabels(yticks)
    ax.set_title(title)
    # Annotate cells with numeric values (format to 2 decimal places)
    for (i, j), val in np.ndenumerate(matrix):
        txt = f"{val:.2f}"
        ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig


def plot_trajectory(epochs: List[int], values: List[float], label: str = "", title: str = "", out_path: Optional[str] = None) -> Figure:
    """Plot a trajectory (e.g., MI over epochs) and return Figure.

    Args:
        xs: x coordinates (e.g., epochs)
        ys: y values
        label: series label
        title: plot title
        out_path: optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, values, marker="o", label=label)
    if label:
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_mi_curves(layers: List[str], mi_in2L: List[float], mi_L2y: List[float], title: str, out_png: str) -> str:
    """Plot two MI curves (input->layer and layer->label) across layers, save PNG, return path."""
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.6), 4))
    x = np.arange(len(layers))
    ax.plot(x, mi_in2L, marker="o", linestyle="-", label="Input->Layer")
    ax.plot(x, mi_L2y, marker="s", linestyle="--", label="Layer->Label")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_mi_heatmap_per_channel(layer_name: str, vec: np.ndarray, title: str, out_png: str) -> str:
    """Plot a heatmap (1 x C) of per-channel MI for a layer, save PNG, return path.

    Args:
        layer_name: name used in x-axis/title
        vec: 1D numpy array of length C
        title: plot title
        out_png: output PNG path
    """
    # Ensure 2D array for imshow: shape (1, C)
    arr = np.atleast_2d(vec.reshape(1, -1))
    fig, ax = plt.subplots(figsize=(max(6, arr.shape[1] * 0.25), 2))
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_xticklabels([str(i) for i in range(arr.shape[1])], rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels([layer_name])
    ax.set_title(title)
    # annotate channel values
    for (i, j), val in np.ndenumerate(arr):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_label_mi_overlay(epochs_list: List[List[int]], mi_series_list: List[List[float]], tags: List[str], title: str = "", out_path: Optional[str] = None) -> Figure:
    """Overlay multiple Label-vs-Layer MI trajectories on the same axes.

    epochs_list: list of epoch lists for each series
    mi_series_list: list of MI value lists (one per series)
    tags: list of legend labels for each series
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    # cycle through a few distinct linestyles/markers for clarity
    markers = ["o", "s", "^", "d", "v", "x"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]
    for i, (epochs, mi_vals, tag) in enumerate(zip(epochs_list, mi_series_list, tags)):
        mk = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        ax.plot(epochs, mi_vals, marker=mk, linestyle=ls, linewidth=1.5, label=tag)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Label-layer MI (nats)")
    ax.set_title(title or "Label-layer MI vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig



