"""Small runtime helpers shared by the training entry points."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch


class _NullScaler:
    """GradScaler-shaped no-op used when AMP is off (CPU or --amp off)."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: Any) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


def amp_tools(mode: str, device: torch.device) -> tuple[Callable[[], Any], Any, bool]:
    """Return ``(autocast_factory, scaler, enabled)``.

    ``mode`` is ``auto`` (AMP on CUDA only), ``on`` or ``off``.  Falls back to
    the pre-2.4 ``torch.cuda.amp`` spelling when the new API is unavailable.
    """

    enabled = device.type == "cuda" and mode in {"auto", "on"}
    if not enabled:
        return nullcontext, _NullScaler(), False
    try:
        scaler = torch.amp.GradScaler("cuda")

        def autocast() -> Any:
            return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

        def autocast() -> Any:
            return torch.cuda.amp.autocast()
    return autocast, scaler, True


def tune_backend(device: torch.device) -> None:
    """cuDNN autotuning pays off here: every 3-D conv sees one fixed shape."""

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


__all__ = ["amp_tools", "tune_backend"]
