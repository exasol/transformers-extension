
class MockSequenceTokenizer:
    def __new__(cls, text: str, return_tensors: str):
        return {'texts': text}

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls