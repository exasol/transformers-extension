from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.filling_mask.mock_filling_mask import \
    MockFillingMaskFactory, MockFillingMaskModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.filling_mask_udf import \
        FillingMask
    from tests.unit_tests.udf_wrapper_params.filling_mask.mock_sequence_tokenizer \
        import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.filling_mask.\
        multiple_model_multiple_batch_multiple_models_per_batch import \
        MultipleModelMultipleBatchMultipleModelsPerBatch as params

    udf = FillingMask(
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
    top_k = 2

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text <mask> 1", top_k)] * data_size + \
                 [(None, "bfs_conn2", "sub_dir2", "model2",
                   "text <mask> 2", top_k)] * data_size + \
                 [(None, "bfs_conn3", "sub_dir3", "model3",
                   "text <mask> 3", top_k)] * data_size + \
                 [(None, "bfs_conn4", "sub_dir4", "model4",
                   "text <mask> 4", top_k)] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text <mask> 1", top_k,
                    "text valid 1", 0.1)] * data_size * top_k + \
                  [("bfs_conn2", "sub_dir2", "model2", "text <mask> 2", top_k,
                    "text valid 2", 0.2)] * data_size * top_k + \
                  [("bfs_conn3", "sub_dir3", "model3", "text <mask> 3", top_k,
                    "text valid 3", 0.3)] * data_size * top_k + \
                  [("bfs_conn4", "sub_dir4", "model4", "text <mask> 4", top_k,
                    "text valid 4", 0.4)] * data_size * top_k

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

    mock_factory = MockFillingMaskFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockFillingMaskModel(sequence="text valid 1", score=0.1),
        PurePosixPath(base_cache_dir2, "sub_dir2", "model2"):
            MockFillingMaskModel(sequence="text valid 2", score=0.2),
        PurePosixPath(base_cache_dir3, "sub_dir3", "model3"):
            MockFillingMaskModel(sequence="text valid 3", score=0.3),
        PurePosixPath(base_cache_dir4, "sub_dir4", "model4"):
            MockFillingMaskModel(sequence="text valid 4", score=0.4)
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

