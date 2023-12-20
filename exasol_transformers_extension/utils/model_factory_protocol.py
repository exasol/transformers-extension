from pathlib import Path
from typing import Protocol, Union, runtime_checkable

import transformers


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """
    Protocol for better type hints.
    """
    def from_pretrained(self, model_name: str, cache_dir: Path, use_auth_token: str) -> transformers.PreTrainedModel:
        pass

    def save_pretrained(self, save_directory: Union[str, Path]):
        pass