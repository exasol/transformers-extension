from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.text_generation.mock_token_generation import \
    MockTextGenerationFactory, MockTextGenerationModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.text_generation_udf import \
        TextGenerationUDF
    from tests.unit_tests.udf_wrapper_params.text_generation. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.text_generation. \
        multiple_return_full_param_single_model_single_batch import \
        MultipleReturnFullParamSingleModelNameSingleBatch as params

    udf = TextGenerationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleReturnFullParamSingleModelNameSingleBatch:
    """
    multiple return_full_text, single model, single batch
    """
    expected_model_counter = 1
    batch_size = 4
    data_size = 2
    max_length = 10
    return_full_text = True
    not_return_full_text = False

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1", "text 1",
                   max_length, return_full_text)] * data_size + \
                 [(None, "bfs_conn1", "sub_dir1", "model1", "text 1",
                   max_length, not_return_full_text)] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text 1", max_length,
                    return_full_text, "text 1 generated" * max_length, None)
                   ] * data_size + \
                  [("bfs_conn1", "sub_dir1", "model1", "text 1", max_length,
                    not_return_full_text, "text 1 generated" * (max_length-1), None)
                   ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")
    }

    mock_factory = MockTextGenerationFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockTextGenerationModel(text_data="text 1")
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
