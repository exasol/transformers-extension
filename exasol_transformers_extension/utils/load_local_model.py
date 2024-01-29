import torch
import transformers.pipelines
from typing import Optional
from pathlib import Path
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol


class LoadLocalModel:
    """
    Class for loading locally saved models and tokenizers. Also stores information regarding the model and pipeline.

    :_pipeline_factory:      a function to create a transformers pipeline
    :task_name:             name of the current task
    :device:                device to be used for pipeline creation
    :_base_model_factory:    a ModelFactoryProtocol for creating the loaded model
    :_tokenizer_factory:     a ModelFactoryProtocol for creating the loaded tokenizer
    """
    def __init__(self,
                 _pipeline_factory,
                 task_name: str,
                 device: str,
                 base_model_factory: ModelFactoryProtocol,
                 tokenizer_factory: ModelFactoryProtocol
                 ):
        self.pipeline_factory = _pipeline_factory
        self.task_name = task_name
        self.device = device
        self._base_model_factory = base_model_factory
        self._tokenizer_factory = tokenizer_factory
        self._loaded_model_key = None

    @property
    def loaded_model_key(self):
        """Get the current loaded_model_key."""
        return self._loaded_model_key

    def load_models(self,
                    model_path: Path,
                    current_model_key: str
                    ) -> transformers.pipelines.Pipeline:
        """
        Loads a locally saved model and tokenizer from "cache_dir / "pretrained" / model_name".
        Returns new pipeline corresponding to the model and task.

        :model_path:            location of the saved model and tokenizer
        :current_model_key:     key of the model to be loaded
        """

        loaded_model = self._base_model_factory.from_pretrained(str(model_path))
        loaded_tokenizer = self._tokenizer_factory.from_pretrained(str(model_path))

        last_created_pipeline = self.pipeline_factory(
            self.task_name,
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device=self.device,
            framework="pt")
        self._loaded_model_key = current_model_key
        return last_created_pipeline

    def clear_device_memory(self):
        """
        Delete models and free device memory
        """
        torch.cuda.empty_cache()

