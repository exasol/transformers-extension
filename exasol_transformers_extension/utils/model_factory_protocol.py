from pathlib import Path
from typing import Protocol, Union, runtime_checkable, Optional

import transformers


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """
    Protocol for better type hints.
    """
    def from_pretrained(self, model_name: str, cache_dir: Optional[Path]=None, use_auth_token: Optional[str]=None) \
            -> transformers.PreTrainedModel:
        """
        Either downloads a model from Huggingface Hub(all parameters required),
        or loads a locally saved model from file (only requires filepath)

        :model_name:        model name, or path to locally saved model files
        :cache_dir:         optional. Path where downloaded model should be cached
        :use_auth_token:    optional. token for Huggingface hub private models
        """
        pass

    def save_pretrained(self, save_directory: Union[str, Path]):
        pass
