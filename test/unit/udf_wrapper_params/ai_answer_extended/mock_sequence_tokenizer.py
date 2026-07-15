class MockSequenceTokenizer:
    @classmethod
    def from_pretrained(cls, model_path, local_files_only=True):
        return cls
