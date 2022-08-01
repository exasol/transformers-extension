import pytest
from unittest.mock import patch, MagicMock
from exasol_transformers_extension.utils import device_management


class MockedTorchLibrary:
    @staticmethod
    def is_available():
        return True


@patch('torch.cuda', MagicMock(return_value=MockedTorchLibrary))
@pytest.mark.parametrize("device_id, expected",
                         [(None, "cpu"), (0, "cuda:0"), (1, "cuda:1")])
def test_getting_torch_device(device_id, expected):
    device = device_management.get_torch_device(device_id)
    device_name = f"{device.type}:{device.index}" \
        if device.index is not None else device.type
    assert device_name == expected
