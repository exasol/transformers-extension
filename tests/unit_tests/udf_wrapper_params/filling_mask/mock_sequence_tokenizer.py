

class MockSequenceTokenizer:

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        cls.mask_token = "valid"
        return cls

