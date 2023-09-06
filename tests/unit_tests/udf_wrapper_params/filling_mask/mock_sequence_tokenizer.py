

class MockSequenceTokenizer:

    @classmethod
    def from_pretrained(cls, model_name, cache_dir, use_auth_token):
        cls.mask_token = "valid"
        return cls

