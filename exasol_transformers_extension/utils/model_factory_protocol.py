from pathlib import Path
from typing import (
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import transformers


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """
    Protocol for better type hints.
    """

    def from_pretrained(
        self,
        model_name: str,
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> transformers.PreTrainedModel:
        """
        Either downloads a model from Huggingface Hub(all parameters required),
        or loads a locally saved model from file (only requires filepath)

        :model_name:        model name, or path to locally saved model files
        :cache_dir:         optional. Path where downloaded model should be cached
        :use_auth_token:    optional. token for Huggingface hub private models
        :revision: (str, optional, defaults to "main") The specific model version to use.
        It can be a branch name, a tag name, or a commit id, since we use a git-based
         system for storing models and other artifacts on huggingface.co,
          so revision can be any identifier allowed by git.

        """

    def save_pretrained(self, save_directory: Union[str, Path]):
        pass
