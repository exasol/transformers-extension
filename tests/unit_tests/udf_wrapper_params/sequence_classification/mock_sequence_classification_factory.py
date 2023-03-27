from pathlib import PurePosixPath
from typing import Dict, List
from dataclasses import dataclass
from tests.unit_tests.udf_wrapper_params.sequence_classification.mock_sequence_tokenizer import \
    MockSequenceTokenizer


@dataclass
class LabelScore:
    label: str
    score: float


class MockSequenceClassificationModel:
    def __init__(self, label_scores: List[LabelScore]):
        self.label_scores = label_scores

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls


class MockSequenceClassificationFactory:

    def __init__(self, mock_models: Dict[PurePosixPath,
                                         MockSequenceClassificationModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        # the cache_dir path already has model_name
        return self.mock_models[cache_dir]


class MockPipeline:

    def __init__(self,
                 task_type: str,
                 model: MockSequenceClassificationModel,
                 tokenizer: MockSequenceTokenizer,
                 device: str,
                 framework: str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework

    def __call__(self, sequences: List[str], **kwargs):
        if "error" in sequences[0]["text"]:
            raise Exception("Error while performing prediction.")

        result = []
        for label_score in self.model.label_scores:
            result.append({"label": label_score.label,
                           "score": label_score.score})
        return [result] * len(sequences)
