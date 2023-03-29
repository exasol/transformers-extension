from pathlib import PurePosixPath
from typing import Dict, List, Union
from tests.unit_tests.udf_wrapper_params.filling_mask.mock_sequence_tokenizer import MockSequenceTokenizer


class MockFillingMaskModel:
    def __init__(self, sequence: str, score: float, rank: int):
        self.result = {"sequence": sequence, "score": score, "rank": rank}

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    def to(self, device):
        self.device = device
        return self


class MockFillingMaskFactory:
    def __init__(self, mock_models: Dict[PurePosixPath, MockFillingMaskModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        # the cache_dir path already has model_name
        return self.mock_models[cache_dir]


class MockPipeline:
    def __init__(self,
                 task_type: str,
                 model: MockFillingMaskModel,
                 tokenizer: MockSequenceTokenizer,
                 device: str,
                 framework: str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework

    def __call__(self, text_data: List[str], top_k: int) -> \
            List[Dict[str, Union[str, float]]]:
        if "error" in text_data[0]:
            raise Exception("Error while performing prediction.")

        input_size = len(text_data)
        single_result = [self.model.result] * top_k
        return [single_result] * input_size if input_size > 1 else single_result

