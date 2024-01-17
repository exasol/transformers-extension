import torch
import transformers.pipelines
from transformers import AutoModel, AutoTokenizer
from pathlib import Path


class LoadLocalModel:
    """
    Class for loading locally saved models and tokenizers. Also stores information regarding the model and pipeline.

    :pipeline:      current model pipeline
    :task_name:     name of the current task
    :device:        device to be used for pipeline creation
    """
    def __init__(self,
                 pipeline,
                 task_name,
                 device
                 ):
        self.pipeline = pipeline
        self.task_name = task_name
        self.device = device
        self.last_loaded_model = None
        self.last_loaded_tokenizer = None
        self.last_loaded_model_key = None

    def load_models(self, model_name: str,
                    current_model_key,
                    cache_dir: Path
                    ) -> transformers.pipelines.Pipeline:
        """
        Loads a locally saved model and tokenizer from "cache_dir / "pretrained" / model_name".
        Returns new pipeline corresponding to the model and task.

        :model_name:            name of the model to be loaded
        :current_model_key:     Key of the model to be loaded
        :cache_dir:             location of the saved model
        """

        self.last_loaded_model = AutoModel.from_pretrained(str(cache_dir / "pretrained" / model_name)) # or do we want to load tokenizer
        self.last_loaded_tokenizer = AutoTokenizer.from_pretrained(str(cache_dir / "pretrained" / model_name))

        last_created_pipeline = self.pipeline(
            self.task_name,
            model=self.last_loaded_model,
            tokenizer=self.last_loaded_tokenizer,
            device=self.device,
            framework="pt")
        self.last_loaded_model_key = current_model_key
        return last_created_pipeline

    def clear_device_memory(self):
        """
        Delete models and free device memory
        """
        self.last_loaded_model = None
        self.last_loaded_tokenizer = None
        torch.cuda.empty_cache()
