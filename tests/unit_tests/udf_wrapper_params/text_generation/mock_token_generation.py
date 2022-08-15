from pathlib import PurePosixPath
from typing import Dict, List, Union
from tests.unit_tests.udf_wrapper_params.text_generation.\
    mock_sequence_tokenizer import MockSequenceTokenizer


class MockTextGenerationModel:
    def __init__(self, text_data: str):
        self.result = {"generated_text": f"{text_data} generated"}

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    def to(self, device):
        self.device = device
        return self


class MockTextGenerationFactory:
    def __init__(self, mock_models: Dict[PurePosixPath,
                                         MockTextGenerationModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        # the cache_dir path already has model_name
        return self.mock_models[cache_dir]


class MockPipeline:
    def __init__(self,
                 task_type: str,
                 model: MockTextGenerationModel,
                 tokenizer: MockSequenceTokenizer,
                 framework: str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.framework = framework

    def __call__(self, text_data: List[str], **kwargs) -> \
            List[Dict[str, Union[str, float]]]:

        len_generated_text = kwargs["max_length"] \
            if kwargs["return_full_text"] else kwargs["max_length"] - 1
        result = {"generated_text":
                      self.model.result["generated_text"] * len_generated_text}
        return [result] * len(text_data)

