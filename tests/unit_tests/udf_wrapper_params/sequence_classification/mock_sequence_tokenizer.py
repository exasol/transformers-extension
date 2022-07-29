from typing import List


class MockSequenceTokenizer:
    def __init__(self, first_texts: List[str], second_texts: List[str] = None,
                return_tensors: str = None):
        self.first_texts = first_texts
        self.second_texts = second_texts

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    def to(self, device_name: str):
        return {'first_texts': self.first_texts}
