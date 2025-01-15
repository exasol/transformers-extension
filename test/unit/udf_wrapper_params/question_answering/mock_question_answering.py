from pathlib import PurePosixPath
from typing import Dict, List, Union, NewType
from test.unit.udf_wrapper_params.question_answering.\
    mock_sequence_tokenizer import MockSequenceTokenizer


class MockQuestionAnsweringModel:
    def __init__(self, answer: str, score: float, rank: int):
        self.result = {"answer": answer, "score": score, "rank": rank}

    @classmethod
    def from_pretrained(cls, model_path):
        return cls


class MockQuestionAnsweringFactory:
    def __init__(self, mock_models: Dict[PurePosixPath,
                                         MockQuestionAnsweringModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_path):
        return self.mock_models[PurePosixPath(model_path)]


class MockPipeline:
    ResultDict = NewType("ResultDict", Dict[str, Union[str, float]])
    counter = 0

    def __init__(self,
                 task: str,
                 model: MockQuestionAnsweringModel,
                 tokenizer: MockSequenceTokenizer,
                 device: str,
                 framework: str):
        self.task_type = task
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.framework = framework
        MockPipeline.counter += 1

    def __call__(self, question: List[str], context: List[str], top_k: int) -> \
            Union[ResultDict, List[ResultDict],  List[List[ResultDict]]]:
        if "error" in context[0]:
            raise Exception("Error while performing prediction.")

        input_size = len(question)
        if input_size == 1 and top_k == 1:
            return self.model.result

        single_result = [self.model.result] * top_k
        return [single_result] * input_size if input_size > 1 else single_result
