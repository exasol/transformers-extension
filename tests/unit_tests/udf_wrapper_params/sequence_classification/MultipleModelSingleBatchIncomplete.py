from tests.unit_tests.udf_wrapper_params.sequence_classification.BaseUDFWrapperParams import \
    BaseUDFWrapperParams


class MultipleModelSingleBatchIncomplete(BaseUDFWrapperParams):
    """
    multiple model, single batch, last batch incomplete

    batch_size = 9
    data_size = 2
    n_model = 4
    """

    def _single_text(self):
        return [f"Test text {str(i+1)}" for i in range(2)]

    def _text_pair(self):
        pass

    def udf_wrapper():
        from exasol_udf_mock_python.udf_context import UDFContext
        from exasol_transformers_extension.udfs. \
            sequence_classification_single_text_udf import \
            SequenceClassificationSingleText
        from tests.unit_tests.udf_wrapper_params.sequence_classification. \
            MockSequenceClassification import MockSequenceClassification
        from tests.unit_tests.udf_wrapper_params.sequence_classification. \
            MockSequenceTokenizer import MockSequenceTokenizer

        udf = SequenceClassificationSingleText(
            exa,
            cache_dir="dummy_cache_dir",
            batch_size=2,
            base_model=MockSequenceClassification(
                model_based_data_size_at_each_batch=[2, 2, 2, 2]),
            tokenizer=MockSequenceTokenizer)

        def run(ctx: UDFContext):
            udf.run(ctx)
