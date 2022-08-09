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
        multiple_bfsconn_single_subdir_single_model_single_batch import \
        MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch as params

    udf = FillingMask(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch:
    """
    multiple bucketfs connection, single subdir, single model, single batch
    """
    batch_size = 4
    data_size = 2
    top_k1 = 3
    top_k2 = 5

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text <mask> 1", top_k1)] * data_size + \
                 [(None, "bfs_conn2", "sub_dir1", "model1",
                   "text <mask> 1", top_k2)] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text <mask> 1", top_k1,
                    "text valid 1", 0.1)] * data_size * top_k1 + \
                  [("bfs_conn2", "sub_dir1", "model1", "text <mask> 1", top_k2,
                    "text valid 2", 0.2)] * data_size * top_k2

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "bfs_conn2": Connection(address=f"file://{base_cache_dir2}")}

    mock_factory = MockFillingMaskFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockFillingMaskModel(sequence="text valid 1", score=0.1),
        PurePosixPath(base_cache_dir2, "sub_dir1", "model1"):
            MockFillingMaskModel(sequence="text valid 2", score=0.2)
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

