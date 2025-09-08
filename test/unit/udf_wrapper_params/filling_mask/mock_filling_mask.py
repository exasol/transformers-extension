from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.filling_mask.mock_sequence_tokenizer import (
    MockSequenceTokenizer,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


class MockFillingMaskModel:
    def __init__(self, sequence: str, score: float, rank: int):
        self.result = {"sequence": sequence, "score": score, "rank": rank}

    def to(self, device):
        self.device = device
        return self

    @classmethod
    def from_pretrained(cls, model_name, cache_dir, use_auth_token):
        return cls


class MockFillingMaskFactory:
    def __init__(self, mock_models: dict[PurePosixPath, MockFillingMaskModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_path):
        # the model_path path already has model_name
        return self.mock_models[PurePosixPath(model_path)]


class MockPipeline:
    counter = 0

    def __init__(
        self,
        task: str,
        model: "MockFillingMaskModel",
        tokenizer: MockSequenceTokenizer,
        device: str,
        framework: str,
    ):
        self.task_type = task
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework
        MockPipeline.counter += 1

    def __call__(
        self, text_data: list[str], top_k: int
    ) -> list[dict[str, Union[str, float]]]:
        if "error" in text_data[0]:
            raise Exception("Error while performing prediction.")

        input_size = len(text_data)
        single_result = [self.model.result] * top_k
        return [single_result] * input_size if input_size > 1 else single_result
