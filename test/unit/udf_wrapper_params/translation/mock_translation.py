from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.text_generation.mock_sequence_tokenizer import (
    MockSequenceTokenizer,
)
from typing import (
    Dict,
    List,
    Union,
)


class MockTranslationModel:
    def __init__(self, text_data: str):
        self.result = text_data

    @classmethod
    def from_pretrained(cls, model_path):
        return cls

    def to(self, device):
        self.device = device
        return self


class MockTranslationFactory:
    def __init__(self, mock_models: Dict[PurePosixPath, MockTranslationModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_path):
        # the cache_dir path already has model_name
        return self.mock_models[PurePosixPath(model_path)]


class MockPipeline:
    counter = 0

    def __init__(
        self,
        task: str,
        model: MockTranslationModel,
        tokenizer: MockSequenceTokenizer,
        device: str,
        framework: str,
    ):
        self.task_type = task
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework
        self.lang_translation = {"German:": "übersetzt", "French:": "traduit"}
        MockPipeline.counter += 1

    def __call__(self, text_data: List[str], **kwargs) -> List[Dict[str, str]]:
        if "error" in text_data[0]:
            raise Exception("Error while performing prediction.")

        max_len = kwargs["max_length"]
        results = []
        for text in text_data:
            splitted_text = text.split()
            target_lang = splitted_text[3]
            translated_text = " ".join(
                (self.model.result, self.lang_translation[target_lang])
            )
            results.append({"translation_text": translated_text * max_len})

        return results
