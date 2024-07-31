import torch
import transformers.pipelines
from typing import Optional
from pathlib import Path

from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.utils.model_specification import ModelSpecification

class LoadLocalModel:
    """
    Class for loading locally saved models and tokenizers. Also stores information regarding the model and pipeline.

    :_pipeline_factory:      a function to create a transformers pipeline
    :task_type:              name of the current task
    :device:                 device to be used for pipeline creation, i.e "CPU"
    :_base_model_factory:    a ModelFactoryProtocol for creating the loaded model
    :_tokenizer_factory:     a ModelFactoryProtocol for creating the loaded tokenizer
    """
    def __init__(self,
                 pipeline_factory,
                 task_type: str,
                 device: str,
                 base_model_factory: ModelFactoryProtocol,
                 tokenizer_factory: ModelFactoryProtocol
                 ):
        self.pipeline_factory = pipeline_factory
        self.task_type = task_type
        self.device = device
        self._base_model_factory = base_model_factory
        self._tokenizer_factory = tokenizer_factory
        self._current_model_specification = None
        self._bucketfs_model_cache_dir = None
        self.last_model_loaded_successfully = None
        self.model_load_error = None

    @property
    def current_model_specification(self):
        """Get the current current_model_specification."""
        return self._current_model_specification

    def set_current_model_specification(self, current_model_specification: BucketFSModelSpecification):
        """Set the current_model_specification."""
        self._current_model_specification = current_model_specification

    def load_models(self) -> transformers.pipelines.Pipeline:
        """
        Loads a locally saved model and tokenizer from model_path.
        Returns new pipeline corresponding to the model and task.

        :model_path:            location of the saved model and tokenizer
        :current_model_key:     key of the model to be loaded
        """

        loaded_model = self._base_model_factory.from_pretrained(str(self._bucketfs_model_cache_dir))
        loaded_tokenizer = self._tokenizer_factory.from_pretrained(str(self._bucketfs_model_cache_dir))

        last_created_pipeline = self.pipeline_factory(
            self.task_type,
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device=self.device,
            framework="pt")
        self.last_model_loaded_successfully = True
        return last_created_pipeline

    def set_bucketfs_model_cache_dir(
            self, bucketfs_location) -> None:
        """
        Set the cache directory in bucketfs of the specified model.
        :param bucketfs_conn_name: Name of the bucketFS connection
        """
        model_path = self._current_model_specification.get_bucketfs_model_save_path()
        self._bucketfs_model_cache_dir = bucketfs_operations.get_local_bucketfs_path(
            bucketfs_location=bucketfs_location, model_path=str(model_path))

    def clear_device_memory(self):
        """
        Delete models and free device memory
        """
        torch.cuda.empty_cache()

