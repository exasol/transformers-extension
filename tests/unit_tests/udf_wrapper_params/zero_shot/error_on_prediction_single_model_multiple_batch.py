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
        error_on_prediction_single_model_multiple_batch import \
        ErrorOnPredictionSingleModelMultipleBatch as params

    udf = ZeroShotTextClassificationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class ErrorOnPredictionSingleModelMultipleBatch:
    """
    not cached error, single model, multiple batch
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 5

    input_data = [(None, "bfs_conn1", "token_conn1", "sub_dir1", "model1",
                   "error on pred", "label1")] * data_size
    output_data = [("bfs_conn1", "token_conn1", "sub_dir1", "model1", "error on pred",
                    "label1", None, None, None, "Traceback")] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "token_conn1": Connection(address='', password="token")
    }

    mock_factory = MockZeroShotFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockZeroShotModel([{"labels": "label1", "scores": 0.1}])
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

