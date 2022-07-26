from typing import List


class MockSequenceTokenizer:
    def __new__(cls, first_texts: List[str], second_texts: List[str] = None,
                return_tensors: str = None):
        return {'first_texts': first_texts}

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls
