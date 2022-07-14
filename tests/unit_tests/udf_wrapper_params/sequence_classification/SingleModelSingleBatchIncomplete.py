from tests.unit_tests.udf_wrapper_params.sequence_classification.BaseUDFWrapperParams import \
    BaseUDFWrapperParams


class SingleModelSingleBatchIncomplete(BaseUDFWrapperParams):
    """
    single model, single batch, batch incomplete

    batch_size = 2
    data_size = 1
    """

    def _single_text(self):
        return ["Test text 1"]

    def _text_pair(self):
        pass

    def udf_wrapper():
        import torch
        from typing import Dict
        from dataclasses import dataclass
        from exasol_udf_mock_python.udf_context import UDFContext
        from exasol_transformers_extension.udfs. \
            sequence_classification_single_text_udf import \
            SequenceClassificationSingleText

        class MockSequenceClassification:
            @dataclass
            class Config:
                id2label: Dict[int, str]

            config = Config(id2label={
                0: 'label_1', 1: 'label_2', 2: 'label_3', 3: 'label_4'})

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            @classmethod
            def from_pretrained(cls, model_name, cache_dir):
                return cls

            @property
            def logits(self) -> torch.FloatTensor:
                return torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        class MockSequenceTokenizer:
            def __new__(cls, text: str, return_tensors: str):
                return {}

            @classmethod
            def from_pretrained(cls, model_name, cache_dir):
                return cls

        udf = SequenceClassificationSingleText(
            exa,
            cache_dir="dummy_cache_dir",
            batch_size=2,
            base_model=MockSequenceClassification,
            tokenizer=MockSequenceTokenizer)

        def run(ctx: UDFContext):
            udf.run(ctx)
