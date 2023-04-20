from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.translation.mock_translation import \
    MockTranslationModel, MockTranslationFactory, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.translation_udf import \
        TranslationUDF
    from tests.unit_tests.udf_wrapper_params.translation. \
        mock_translation_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.translation. \
        error_not_cached_single_model_multiple_batch import \
        ErrorNotCachedSingleModelMultipleBatch as params

    udf = TranslationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class ErrorNotCachedSingleModelMultipleBatch:
    """
    not cached error, single model, multiple batch
    """
    expected_model_counter = 0
    batch_size = 2
    data_size = 5
    src_lang = "English"
    target_lang = "German"
    max_length = 10

    input_data = [(None, "bfs_conn1", "sub_dir1", "non_existing_model",
                   "text 1", src_lang, target_lang, max_length)
                  ] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "non_existing_model", "text 1",
                    src_lang, target_lang,  max_length, None, "Traceback")
                   ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockTranslationFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockTranslationModel(text_data="text 1")
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
