from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.\
    mock_token_classification import \
    MockTokenClassificationFactory, MockTokenClassificationModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.token_classification_udf import \
        TokenClassificationUDF
    from tests.unit_tests.udf_wrapper_params.token_classification. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.token_classification.\
        multiple_bfsconn_single_subdir_single_model_multiple_batch import \
        MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch as params

    udf = TokenClassificationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch:
    """
    multiple bucketfs connection, single subdir, single model, multiple_batch
    """
    expected_model_counter = 2
    batch_size = 2
    data_size = 2
    n_entities = 3
    agg_strategy = "simple"

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text1", agg_strategy)] * data_size + \
                 [(None, "bfs_conn2", "sub_dir1", "model1",
                   "text1", agg_strategy)] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text1", agg_strategy,
                    0, 6, "text1", "label1", 0.1, None)
                   ] * n_entities * data_size + \
                  [("bfs_conn2", "sub_dir1", "model1", "text1", agg_strategy,
                    0, 6, "text1", "label2", 0.2, None)
                   ] * n_entities * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")

    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "bfs_conn2": Connection(address=f"file://{base_cache_dir2}")
    }
    mock_factory = MockTokenClassificationFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockTokenClassificationModel(
                starts=[0] * n_entities,
                ends=[6] * n_entities,
                words=["text1"] * n_entities,
                entities=["label1"] * n_entities,
                scores=[0.1] * n_entities),
        PurePosixPath(base_cache_dir2, "sub_dir1", "model1"):
            MockTokenClassificationModel(
                starts=[0] * n_entities,
                ends=[6] * n_entities,
                words=["text1"] * n_entities,
                entities=["label2"] * n_entities,
                scores=[0.2] * n_entities),
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

