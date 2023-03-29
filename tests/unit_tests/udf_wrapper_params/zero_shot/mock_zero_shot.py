from pathlib import PurePosixPath
from typing import Dict, List, Union
from tests.unit_tests.udf_wrapper_params.filling_mask.mock_sequence_tokenizer import MockSequenceTokenizer


class MockZeroShotModel:
    def __init__(self, result: List[Dict]):
        self.result = result

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    def to(self, device):
        self.device = device
        return self


class MockZeroShotFactory:
    def __init__(self, mock_models: Dict[PurePosixPath, MockZeroShotModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        # the cache_dir path already has model_name
        return self.mock_models[cache_dir]


class MockPipeline:
    def __init__(self,
                 task_type: str,
                 model: MockZeroShotModel,
                 tokenizer: MockSequenceTokenizer,
                 device : str,
                 framework: str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework

    def __call__(self, text_data: List[str], labels: List[str]) -> \
            List[List[Dict[str, Union[str, float]]]]:
        if "error" in text_data[0]:
            raise Exception("Error while performing prediction.")

        input_size = len(text_data)
        return [self.model.result] * input_size

