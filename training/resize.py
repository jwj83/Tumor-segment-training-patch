"""Shape-safe 3-D resizing used by training and model input adapters.

Images use trilinear interpolation; masks always use nearest-neighbour.  The
old code resized 2-D slices independently in one training path and used a
different interpolation rule in inference.  This module gives both paths one
explicit implementation and preserves channel/volume dimensions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F


def resize_volume(
    array: np.ndarray | torch.Tensor,
    target_shape: Sequence[int],
    *,
    is_mask: bool = False,
) -> np.ndarray | torch.Tensor:
    """Resize a 3-D or channel-first 4-D volume without changing its type."""

    shape = tuple(int(x) for x in target_shape)
    if len(shape) != 3 or any(x < 1 for x in shape):
        raise ValueError(f"target_shape must contain three positive integers, got {shape}")
    original_is_numpy = isinstance(array, np.ndarray)
    tensor = torch.as_tensor(array)
    if tensor.ndim == 3:
        tensor = tensor[None, None]
        restore = "3d"
    elif tensor.ndim == 4:
        tensor = tensor[None]
        restore = "4d"
    elif tensor.ndim == 5:
        restore = "5d"
    else:
        raise ValueError(f"volume must be 3-D, channel-first 4-D, or 5-D, got {tensor.shape}")
    tensor = tensor.float()
    mode = "nearest" if is_mask else "trilinear"
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    result = F.interpolate(tensor, size=shape, mode=mode, **kwargs)
    if is_mask:
        result = (result >= 0.5).to(torch.uint8)
    if restore == "3d":
        result = result[0, 0]
    elif restore == "4d":
        result = result[0]
    if original_is_numpy:
        return result.cpu().numpy()
    return result


__all__ = ["resize_volume"]
