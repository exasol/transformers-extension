from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_complete_extended.mock_sequence_tokenizer import (
    MockSequenceTokenizer,
)
from typing import (
    Dict,
    List,
    Union,
)


class MockTextGenerationModel:
    def __init__(self, text_data: str):
        self.result = {"generated_text": f"{text_data} generated"}

    @classmethod
    def from_pretrained(cls, model_path):
        return cls

    def to(self, device):
        self.device = device
        return self


class MockTextGenerationFactory:
    def __init__(self, mock_models: dict[PurePosixPath, MockTextGenerationModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_path):
        # the cache_dir path already has model_name
        return self.mock_models[PurePosixPath(model_path)]


class MockPipeline:
    counter = 0

    def __init__(
        self,
        task: str,
        model: MockTextGenerationModel,
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
        self, text_data: list[str], **kwargs
    ) -> list[dict[str, Union[str, float]]]:
        if "error" in text_data[0]:
            raise Exception("Error while performing prediction.")

        len_generated_text = (
            kwargs["max_new_tokens"]
            if kwargs["return_full_text"]
            else kwargs["max_new_tokens"] - 1
        )
        result = {
            "generated_text": self.model.result["generated_text"] * len_generated_text
        }
        return [result] * len(text_data)
