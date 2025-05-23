"""Function get_torch_device to set the device(CPU/CUDA device) PyTorch will use"""

from typing import Optional

import torch


def get_torch_device(device_id: Optional[int]) -> torch.device:
    """
    Set torch device to push models

    :param device_id: Either the id of cuda device or None implying CPU
    """
    device_name = (
        f"cuda:{device_id}"
        if torch.cuda.is_available() and device_id is not None
        else "cpu"
    )
    return torch.device(device_name)
