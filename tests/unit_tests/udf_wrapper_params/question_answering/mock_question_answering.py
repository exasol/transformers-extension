from typing import Dict, List, Union
from tests.unit_tests.udf_wrapper_params.question_answering.\
    mock_sequence_tokenizer import MockSequenceTokenizer


class MockQuestionAnsweringModel:
    def __init__(self, answer: str, score: float):
        self.result = {"answer": answer, "score": score}

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls


class MockQuestionAnsweringFactory:
    def __init__(self, mock_models: Dict[str, MockQuestionAnsweringModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        return self.mock_models[model_name]


class MockPipeline:
    def __init__(self,
                 task_type: str,
                 model: MockQuestionAnsweringModel,
                 tokenizer: MockSequenceTokenizer,
                 device : str):
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, question: List[str], context: List[str]) -> \
            List[Dict[str, Union[str, float]]]:
        return [self.model.result] * len(question)

