from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.text_generation.mock_token_generation import (
    MockPipeline,
    MockTextGenerationFactory,
    MockTextGenerationModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.text_generation.error_on_prediction_multiple_model_multiple_batch import (
        ErrorOnPredictionMultipleModelMultipleBatch as params,
    )
    from test.unit.udf_wrapper_params.text_generation.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.text_generation_udf import (
        TextGenerationUDF,
    )

    udf = TextGenerationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


class ErrorOnPredictionMultipleModelMultipleBatch:
    """
    not cached error, multiple model, multiple batch
    """

    expected_model_counter = 2
    batch_size = 3
    data_size = 2
    max_length = 10
    return_full_text = True

    input_data = [
        (
            None,
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            max_length,
            return_full_text,
        )
    ] * data_size + [
        (
            None,
            "bfs_conn2",
            "sub_dir2",
            "model2",
            "error on pred",
            max_length,
            return_full_text,
        )
    ] * data_size
    output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            max_length,
            return_full_text,
            "text 1 generated" * max_length,
            None,
        )
    ] * data_size + [
        (
            "bfs_conn2",
            "sub_dir2",
            "model2",
            "error on pred",
            max_length,
            return_full_text,
            None,
            "Traceback",
        )
    ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "bfs_conn2": Connection(address=f"file://{base_cache_dir2}"),
    }
    mock_factory = MockTextGenerationFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_text-generation"
            ): MockTextGenerationModel(text_data="text 1"),
            PurePosixPath(
                base_cache_dir2, "sub_dir2", "model2_text-generation"
            ): MockTextGenerationModel(text_data="text 2"),
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
