from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.text_generation.mock_token_generation import \
    MockTextGenerationFactory, MockTextGenerationModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.text_generation_udf import \
        TextGeneration
    from tests.unit_tests.udf_wrapper_params.text_generation. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.text_generation. \
        multiple_model_multiple_batch_multiple_models_per_batch import \
        MultipleModelMultipleBatchMultipleModelsPerBatch as params

    udf = TextGeneration(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleModelMultipleBatchMultipleModelsPerBatch:
    """
    multiple model, multiple batch, multiple models per batch
    """
    batch_size = 2
    data_size = 1
    max_length = 10
    return_full_text = True

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1", "text 1",
                   max_length, return_full_text)] * data_size + \
                 [(None, "bfs_conn2", "sub_dir2", "model2", "text 2",
                   max_length, return_full_text)] * data_size + \
                 [(None, "bfs_conn3", "sub_dir3", "model3", "text 3",
                   max_length, return_full_text)] * data_size + \
                 [(None, "bfs_conn4", "sub_dir4", "model4", "text 4",
                   max_length, return_full_text)] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text 1", max_length,
                    return_full_text, "text 1 generated" * max_length)
                   ] * data_size + \
                  [("bfs_conn2", "sub_dir2", "model2", "text 2", max_length,
                    return_full_text, "text 2 generated" * max_length)
                   ] * data_size + \
                  [("bfs_conn3", "sub_dir3", "model3", "text 3", max_length,
                    return_full_text, "text 3 generated" * max_length)
                   ] * data_size + \
                  [("bfs_conn4", "sub_dir4", "model4", "text 4", max_length,
                    return_full_text, "text 4 generated" * max_length)
                   ] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")
    base_cache_dir3 = PurePosixPath(tmpdir_name, "bfs_conn3")
    base_cache_dir4 = PurePosixPath(tmpdir_name, "bfs_conn4")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "bfs_conn2": Connection(address=f"file://{base_cache_dir2}"),
        "bfs_conn3": Connection(address=f"file://{base_cache_dir3}"),
        "bfs_conn4": Connection(address=f"file://{base_cache_dir4}")}

    mock_factory = MockTextGenerationFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockTextGenerationModel(text_data="text 1"),
        PurePosixPath(base_cache_dir2, "sub_dir2", "model2"):
            MockTextGenerationModel(text_data="text 2"),
        PurePosixPath(base_cache_dir3, "sub_dir3", "model3"):
            MockTextGenerationModel(text_data="text 3"),
        PurePosixPath(base_cache_dir4, "sub_dir4", "model4"):
            MockTextGenerationModel(text_data="text 4")
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
