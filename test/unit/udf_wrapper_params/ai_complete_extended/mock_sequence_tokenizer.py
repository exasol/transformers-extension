class MockSequenceTokenizer:
    @classmethod
    def from_pretrained(cls, model_name, local_files_only=True):
        return cls
