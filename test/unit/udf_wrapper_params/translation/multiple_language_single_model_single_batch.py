from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.translation.mock_translation import (
    MockPipeline,
    MockTranslationFactory,
    MockTranslationModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.translation.mock_translation_tokenizer import (
        MockSequenceTokenizer,
    )
    from test.unit.udf_wrapper_params.translation.multiple_language_single_model_single_batch import (
        MultipleLanguageSingleModelNameSingleBatch as params,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.translation_udf import TranslationUDF

    udf = TranslationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleLanguageSingleModelNameSingleBatch:
    """
    multiple language, single model, single batch
    """

    expected_model_counter = 1
    batch_size = 4
    data_size = 2
    max_length = 10
    src_lang = "English"
    target_lang1 = "German"
    target_lang2 = "French"

    input_data = [
        (
            None,
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            src_lang,
            target_lang1,
            max_length,
        )
    ] * data_size + [
        (
            None,
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            src_lang,
            target_lang2,
            max_length,
        )
    ] * data_size
    output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            src_lang,
            target_lang1,
            max_length,
            "text 1 übersetzt" * max_length,
            None,
        )
    ] * data_size + [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            src_lang,
            target_lang2,
            max_length,
            "text 1 traduit" * max_length,
            None,
        )
    ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockTranslationFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_translation"
            ): MockTranslationModel(text_data="text 1")
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
