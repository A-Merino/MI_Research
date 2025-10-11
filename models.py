"""models — Backbone builders (VGG / ResNet) with notebook-compatible heads

Convenience builders for VGG16, ResNet18, and ResNet50 that replace final
classifier layers to match the notebook's training heads.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18, resnet50, VGG16_Weights, ResNet18_Weights, ResNet50_Weights
import logging

logger = logging.getLogger(__name__)


def activation_factory(name: str):
    """Return an activation module for a short name.

    Supported: 'relu', 'tanh'.
    """
    name = name.lower()
    if name == "relu":
        return torch.nn.ReLU(inplace=True)
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def replace_activations(module: torch.nn.Module, from_cls, to_mod_factory):
    """Recursively replace activation modules of type ``from_cls`` with
    new modules produced by ``to_mod_factory()``.

    This walks the module tree and replaces children in-place.
    """
    for n, child in list(module.named_children()):
        if isinstance(child, from_cls):
            setattr(module, n, to_mod_factory())
        else:
            replace_activations(child, from_cls, to_mod_factory)


def build_vgg16(pretrained: bool, num_classes: int, activation: str = "relu", use_dropout: bool = True, p_dropout: float = 0.5) -> nn.Module:
    """Build VGG16: keep convolutional body, replace classifier with a
    Dropout+ReLU head compatible with the notebook.

    The replacement head assumes the VGG feature map for 224x224 inputs
    (512*7*7 -> 25088 features) and creates:
        Flatten -> Linear(25088, 4096) -> ReLU -> Dropout(0.5) -> Linear(4096, num_classes)

    Args:
        pretrained: whether to load pretrained weights for the base VGG16.
        num_classes: number of output classes.
        activation: 'relu' (default) or 'tanh'.

    Returns:
        A nn.Module (VGG) with a custom classifier head.
    """

    model = vgg16(weights=(VGG16_Weights.DEFAULT if pretrained else None))

    # If a non-default activation is requested, replace ReLU modules
    if activation != "relu":
        replace_activations(model, torch.nn.ReLU, lambda: activation_factory(activation))

    # Determine expected flattened feature size for classifier
    # For standard VGG with 224x224 input -> 512 * 7 * 7 = 25088
    in_features = model.classifier[0].in_features if hasattr(model, 'classifier') else 25088

    # New classifier head matching the notebook: Flatten -> 256 -> 256 -> num_classes
    # Optionally include Dropout between the two hidden ReLU layers.
    layers = []
    layers.append(nn.Linear(in_features, 256))
    layers.append(nn.ReLU(inplace=True))
    if use_dropout:
        layers.append(nn.Dropout(p=p_dropout))
    layers.append(nn.Linear(256, 256))
    layers.append(nn.ReLU(inplace=True))
    if use_dropout:
        layers.append(nn.Dropout(p=p_dropout))
    layers.append(nn.Linear(256, num_classes))

    classifier = nn.Sequential(*layers)

    model.classifier = classifier
    return model


def build_resnet18(pretrained: bool, num_classes: int, activation: str = "relu") -> nn.Module:
    """Build ResNet18 and replace the final fully-connected layer.

    Args:
        pretrained: whether to load pretrained weights.
        num_classes: number of output classes.
        activation: 'relu' (default) or 'tanh'.
    """

    model = resnet18(weights=(ResNet18_Weights.DEFAULT if pretrained else None))
    if activation != "relu":
        replace_activations(model, torch.nn.ReLU, lambda: activation_factory(activation))
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(pretrained: bool, num_classes: int, activation: str = "relu") -> nn.Module:
    """Build ResNet50 and replace the final fully-connected layer.

    Args:
        pretrained: whether to load pretrained weights.
        num_classes: number of output classes.
        activation: 'relu' (default) or 'tanh'.
    """

    model = resnet50(weights=(ResNet50_Weights.DEFAULT if pretrained else None))
    if activation != "relu":
        replace_activations(model, torch.nn.ReLU, lambda: activation_factory(activation))
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze backbone parameters.

    For VGG-style models this will freeze `model.features` parameters. For
    ResNet-style models this will freeze all parameters except the final
    classifier (`model.fc`).
    """

    if hasattr(model, 'features'):
        # VGG-like
        for p in model.features.parameters():
            p.requires_grad = not freeze
    else:
        # Default: freeze all then unfreeze classifier if present
        for p in model.parameters():
            p.requires_grad = not freeze
        if hasattr(model, 'fc') and freeze:
            # if freezing backbone we still want classifier params trainable
            for p in model.fc.parameters():
                p.requires_grad = True


def load_partial_state_dict(model: nn.Module, state: dict, strict: bool = False) -> dict:
    """Load keys from `state` into `model` when names exist and shapes match.

    This helper is tolerant to mismatched classifier sizes (common when
    loading a checkpoint trained for a different number of classes).

    Args:
        model: target model to load weights into.
        state: state dict or checkpoint containing a 'state_dict' mapping.
        strict: if True, will raise when important mismatches occur; for
            normal tolerant loads set to False.

    Returns:
        A dict summarizing loaded keys and mismatch lists: {"loaded": [...],
        "missing": [...], "mismatched": [...], "unexpected": [...]}.
    """
    # accept full checkpoint dicts that contain 'state_dict'
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        source = state["state_dict"]
    else:
        source = state

    model_state = model.state_dict()
    to_load = {}
    loaded_keys = []
    mismatched = []
    for k, v in source.items():
        if k in model_state:
            if isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
                to_load[k] = v
                loaded_keys.append(k)
            else:
                mismatched.append(k)
        else:
            # unexpected key (e.g., module prefix differences)
            # we'll ignore here
            pass

    # perform the partial load
    if to_load:
        msg = f"Loading {len(to_load)} matching tensors into model ({len(model_state)} params)"
        logger.info(msg)
        model.load_state_dict(to_load, strict=False)
    else:
        logger.info("No matching keys found to load into model")

    missing = [k for k in model_state.keys() if k not in to_load]
    unexpected = [k for k in source.keys() if k not in model_state]

    result = {"loaded": loaded_keys, "missing": missing, "mismatched": mismatched, "unexpected": unexpected}

    if strict and (mismatched or unexpected):
        raise RuntimeError(f"Partial load encountered mismatches: {result}")

    # log summary at debug/info levels
    if mismatched:
        logger.warning(f"Mismatched keys (name found but shape mismatch): {mismatched}")
    if unexpected:
        logger.info(f"Unexpected keys in checkpoint not present in model: {len(unexpected)}")

    return result



