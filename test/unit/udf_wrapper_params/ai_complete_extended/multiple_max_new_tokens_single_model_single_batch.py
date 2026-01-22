from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_complete_extended.mock_token_generation import (
    MockPipeline,
    MockTextGenerationFactory,
    MockTextGenerationModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.ai_complete_extended.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )
    from test.unit.udf_wrapper_params.ai_complete_extended.multiple_max_new_tokens_single_model_single_batch import (
        MultipleMaxLengthSingleModelNameSingleBatch as params,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.ai_complete_extended_udf import (
        AiCompleteExtendedUDF,
    )

    udf = AiCompleteExtendedUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleMaxLengthSingleModelNameSingleBatch:
    """
    multiple max_new_tokens, single model, single batch
    """

    expected_model_counter = 1
    batch_size = 4
    data_size = 2
    max_new_tokens1 = 10
    max_new_tokens2 = 20
    return_full_text = True

    input_data = [
        (
            None,
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            max_new_tokens1,
            return_full_text,
        )
    ] * data_size + [
        (
            None,
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            max_new_tokens2,
            return_full_text,
        )
    ] * data_size
    output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            max_new_tokens1,
            return_full_text,
            "text 1 generated" * max_new_tokens1,
            None,
        )
    ] * data_size + [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "text 1",
            max_new_tokens2,
            return_full_text,
            "text 1 generated" * max_new_tokens2,
            None,
        )
    ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockTextGenerationFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_text-generation"
            ): MockTextGenerationModel(text_data="text 1")
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
