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
        single_model_single_batch_complete import \
        SingleModelSingleBatchComplete as params

    udf = TokenClassificationUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class SingleModelSingleBatchComplete:
    """
    single model, single batch, batch complete
    """
    expected_model_counter = 1
    batch_size = 1
    data_size = 1
    n_entities = 3
    agg_strategy = "simple"

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "text", agg_strategy)] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text", agg_strategy,
                    0, 6, "text", "label1", 0.1, None)] * n_entities * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")
    }

    mock_factory = MockTokenClassificationFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockTokenClassificationModel(
                starts=[0] * n_entities,
                ends=[6] * n_entities,
                words=["text"]*n_entities,
                entities=["label1"]*n_entities,
                scores=[0.1]*n_entities,
                token_spans=["(0,6)"] * n_entities),
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

