import copy
from pathlib import PurePosixPath
from typing import Dict, List, Union
from tests.unit_tests.udf_wrapper_params.token_classification.\
    mock_sequence_tokenizer import MockSequenceTokenizer


class MockTokenClassificationModel:
    def __init__(self, starts: List[int], ends: List[int], words: List[str],
                 entities: List[str], scores: List[float]):
        self.result = [{"start": start, "end": end, "word": word,
                        "entity_group": entity, "score": score}
                       for start, end, word, entity, score
                       in zip(starts, ends, words, entities, scores)]

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
                 device : str,
                 framework: str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework

    def __call__(self, text_data: List[str], aggregation_strategy: str) -> \
            List[Dict[str, Union[str, float]]]:
        if "error" in text_data[0]:
            raise Exception("Error while performing prediction.")

        result_list = self._get_result_list(aggregation_strategy)
        return [result_list] * len(text_data) \
            if len(text_data) > 1 else result_list

    def _get_result_list(self, aggregation_strategy: str):
        result_list = copy.deepcopy(self.model.result)
        if aggregation_strategy == "none":
            for i, result in enumerate(result_list):
                result["entity"] = result.pop("entity_group")
                result_list[i] = result
        return result_list

