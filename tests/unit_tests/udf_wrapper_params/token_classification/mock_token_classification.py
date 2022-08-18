from pathlib import PurePosixPath
from typing import Dict, List, Union
from tests.unit_tests.udf_wrapper_params.token_classification.\
    mock_sequence_tokenizer import MockSequenceTokenizer


class MockTokenClassificationModel:
    def __init__(self, indexes: List[int], words: List[str],
                 entities: List[str], scores: List[float]):
        self.result = [{"index": index, "word": word,
                        "entity": entity, "score": score}
                       for index, word, entity, score
                       in zip(indexes, words, entities, scores)]

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    def to(self, device):
        self.device = device
        return self


class MockTokenClassificationFactory:
    def __init__(self, mock_models: Dict[PurePosixPath,
                                         MockTokenClassificationModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        # the cache_dir path already has model_name
        return self.mock_models[cache_dir]


class MockPipeline:
    def __init__(self,
                 task_type: str,
                 model: MockTokenClassificationModel,
                 tokenizer: MockSequenceTokenizer,
                 framework: str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.framework = framework

    def __call__(self, text_data: List[str]) -> \
            List[Dict[str, Union[str, float]]]:
        return [self.model.result] * len(text_data) if len(text_data) > 1 \
            else self.model.result

