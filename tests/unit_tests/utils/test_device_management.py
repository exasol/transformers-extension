from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from exasol_transformers_extension.utils import device_management


@patch("torch.cuda.is_available", MagicMock(return_value=True))
@pytest.mark.parametrize(
    "device_id, expected", [(None, "cpu"), (0, "cuda:0"), (1, "cuda:1")]
)
def test_getting_torch_device_available(device_id, expected):
    device = device_management.get_torch_device(device_id)
    device_name = (
        f"{device.type}:{device.index}" if device.index is not None else device.type
    )

    assert device_name == expected


@patch("torch.cuda.is_available", MagicMock(return_value=False))
@pytest.mark.parametrize("device_id, expected", [(None, "cpu"), (0, "cpu"), (1, "cpu")])
def test_getting_torch_device_not_available(device_id, expected):
    device = device_management.get_torch_device(device_id)
    device_name = device.type

    assert device_name == expected
