from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.zero_shot.mock_zero_shot import \
    MockZeroShotFactory, MockZeroShotModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.\
        zero_shot_text_classification_udf import ZeroShotTextClassificationUDF
    from tests.unit_tests.udf_wrapper_params.zero_shot.mock_sequence_tokenizer \
        import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.zero_shot.\
        multiple_model_multiple_batch_multiple_model_per_batch import \
        MultipleModelMultipleBatchMultipleModelsPerBatch as params

    udf = ZeroShotTextClassificationUDF(
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

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text1", "label1")] * data_size + \
                 [(None, "bfs_conn2", "sub_dir2", "model2",
                   "text2", "label2")] * data_size + \
                 [(None, "bfs_conn3", "sub_dir3", "model3",
                   "text3", "label3")] * data_size + \
                 [(None, "bfs_conn4", "sub_dir4", "model4",
                   "text4", "label4")] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text1",
                    "label1", "label1", 0.1, 1)] * data_size + \
                  [("bfs_conn2", "sub_dir2", "model2", "text2",
                    "label2", "label2", 0.2, 1)] * data_size + \
                  [("bfs_conn3", "sub_dir3", "model3", "text3",
                    "label3", "label3", 0.3, 1)] * data_size + \
                  [("bfs_conn4", "sub_dir4", "model4", "text4",
                    "label4", "label4", 0.4, 1)] * data_size

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

    mock_factory = MockZeroShotFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockZeroShotModel([{"labels": "label1", "scores": 0.1}]),
        PurePosixPath(base_cache_dir2, "sub_dir2", "model2"):
            MockZeroShotModel([{"labels": "label2", "scores": 0.2}]),
        PurePosixPath(base_cache_dir3, "sub_dir3", "model3"):
            MockZeroShotModel([{"labels": "label3", "scores": 0.3}]),
        PurePosixPath(base_cache_dir4, "sub_dir4", "model4"):
            MockZeroShotModel([{"labels": "label4", "scores": 0.4}])
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

