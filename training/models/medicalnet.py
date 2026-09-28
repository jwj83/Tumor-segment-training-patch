"""Optional MedicalNet adapters used by training and competition tasks.

MedicalNet is kept as an optional dependency because the competition image and
weight packages are usually mounted separately.  The adapter accepts both the
official ``models.resnet.resnet50`` API and classifier-style forks, and accepts
raw state dicts plus common ``{"state_dict": ...}`` checkpoint formats.  It
never downloads weights.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("MedicalNet checkpoint must be a state_dict or a checkpoint dict")
    return {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _medicalnet_factory(depth: int, *, in_channels: int, num_classes: int) -> tuple[nn.Module, str]:
    """Build either the newer classifier-style or official MedicalNet model.

    The public MedicalNet repository exposes ``resnet50``/``resnet101`` in
    ``models.resnet``; it does not expose the classifier-style ``generate_model``
    function used by some forks.  Supporting both keeps the training code
    compatible with the mounted public model package.
    """
    errors: list[str] = []
    for module_name in ("models.resnet", "medicalnet.models.resnet"):
        try:
            module = importlib.import_module(module_name)
            generate_model = getattr(module, "generate_model", None)
            if generate_model is not None:
                model = generate_model(
                    model_depth=depth,
                    n_classes=num_classes,
                    n_input_channels=in_channels,
                    shortcut_type="B",
                    conv1_t_size=7,
                    conv1_t_stride=1,
                    no_max_pool=False,
                )
                return model, "classifier_api"
            factory = getattr(module, f"resnet{depth}", None)
            if factory is not None:
                if in_channels != 1:
                    raise ValueError(
                        "The official MedicalNet ResNet implementation has a fixed "
                        "single-channel input; use in_channels=1."
                    )
                model = factory(
                    sample_input_W=32,
                    sample_input_H=32,
                    sample_input_D=16,
                    shortcut_type="B",
                    no_cuda=True,
                    num_seg_classes=num_classes,
                )
                return model, "official_api"
            errors.append(f"{module_name}: no supported MedicalNet factory")
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "MedicalNet is not importable. Add the MedicalNet repository to PYTHONPATH "
        "or install its package before selecting backend=medicalnet. "
        + " | ".join(errors)
    )


def _load_compatible(module: nn.Module, checkpoint: str | Path, *, strict: bool) -> dict[str, Any]:
    raw = _unwrap_state_dict(torch.load(checkpoint, map_location="cpu"))
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        key = key.removeprefix("backbone.")
        normalized[key] = value
    target = module.state_dict()
    compatible = {
        key: value
        for key, value in normalized.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    mismatched = sorted(
        key for key, value in normalized.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    )
    missing = sorted(set(target) - set(compatible))
    unexpected = sorted(set(normalized) - set(target))
    module.load_state_dict(compatible, strict=False)
    report = {
        "path": str(checkpoint),
        "loaded_keys": len(compatible),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
    }
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(f"MedicalNet checkpoint is not compatible: {report}")
    return report


class _OfficialMedicalNetFeatures(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        last_block = backbone.layer4[-1]
        if hasattr(last_block, "conv3"):
            self.feature_dim = int(last_block.conv3.out_channels)
        elif hasattr(last_block, "conv2"):
            self.feature_dim = int(last_block.conv2.out_channels)
        else:
            raise RuntimeError("Cannot determine MedicalNet feature dimension")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)


class MedicalNet3DClassifier(nn.Module):
    """MedicalNet ResNet classifier for ``(B, C, D, H, W)`` inputs."""

    def __init__(
        self,
        *,
        depth: int = 50,
        in_channels: int = 4,
        num_classes: int = 1,
        checkpoint: str | Path | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        backbone, self._api = _medicalnet_factory(
            depth, in_channels=in_channels, num_classes=num_classes
        )
        self.checkpoint_report: dict[str, Any] = {}
        if checkpoint:
            self.checkpoint_report = _load_compatible(
                backbone, checkpoint, strict=strict
            )
        if self._api == "official_api":
            self.encoder = _OfficialMedicalNetFeatures(backbone)
            self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        else:
            self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._api == "official_api":
            return self.classifier(self.encoder(x))
        output = self.backbone(x)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        return output


class MedicalNet3DEncoder(nn.Module):
    """MedicalNet backbone with its final classifier replaced by Identity."""

    def __init__(
        self,
        *,
        depth: int = 50,
        in_channels: int = 4,
        checkpoint: str | Path | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, api = _medicalnet_factory(
            depth, in_channels=in_channels, num_classes=1
        )
        self.checkpoint_report: dict[str, Any] = {}
        if checkpoint:
            self.checkpoint_report = _load_compatible(
                self.backbone, checkpoint, strict=strict
            )
        if api == "official_api":
            self.encoder = _OfficialMedicalNetFeatures(self.backbone)
            self.feature_dim = self.encoder.feature_dim
        else:
            classifier = getattr(self.backbone, "fc", None)
            if classifier is None or not hasattr(classifier, "in_features"):
                raise RuntimeError("MedicalNet backbone does not expose a final fc layer")
            self.feature_dim = int(classifier.in_features)
            self.backbone.fc = nn.Identity()
            self.encoder = self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).reshape(x.shape[0], -1)


__all__ = ["MedicalNet3DClassifier", "MedicalNet3DEncoder"]
