

class MockSequenceTokenizer:

    @classmethod
    def from_pretrained(cls, model_path):
        cls.mask_token = "valid"
        return cls
