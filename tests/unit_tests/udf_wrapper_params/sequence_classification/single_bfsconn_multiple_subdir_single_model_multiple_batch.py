from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.sequence_classification. \
    mock_sequence_classification_factory import \
    LabelScore, MockSequenceClassificationFactory, \
    MockSequenceClassificationModel, MockPipeline


def udf_wrapper_single_text():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.sequence_classification_single_text_udf import \
        SequenceClassificationSingleTextUDF
    from tests.unit_tests.udf_wrapper_params.sequence_classification. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.sequence_classification. \
        single_bfsconn_multiple_subdir_single_model_multiple_batch import \
        SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch as params

    udf = SequenceClassificationSingleTextUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


def udf_wrapper_text_pair():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.sequence_classification_text_pair_udf import \
        SequenceClassificationTextPairUDF
    from tests.unit_tests.udf_wrapper_params.sequence_classification. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.sequence_classification.\
        single_bfsconn_multiple_subdir_single_model_multiple_batch import \
        SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch as params

    udf = SequenceClassificationTextPairUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch:
    """
    single bucketfs connection, multiple subdir, single model, multiple batch
    """
    expected_single_text_model_counter = 2
    expected_text_pair_model_counter = 2
    batch_size = 2
    data_size = 2

    label_scores1 = [
        LabelScore('label1', 0.21),
        LabelScore('label2', 0.24),
        LabelScore('label3', 0.26),
        LabelScore('label4', 0.29),
    ]
    label_scores2 = [
        LabelScore('label1', 0.25),
        LabelScore('label2', 0.25),
        LabelScore('label3', 0.25),
        LabelScore('label4', 0.25),
    ]

    inputs_single_text = [(None, "bfs_conn1", "sub_dir1",
                           "model1", "My test text")] * data_size + \
                         [(None, "bfs_conn1", "sub_dir2",
                           "model1", "My test text")] * data_size
    inputs_pair_text = [(None, "bfs_conn1", "sub_dir1", "model1",
                         "My text 1", "My text 2")] * data_size + \
                       [(None, "bfs_conn1", "sub_dir2", "model1",
                         "My text 1", "My text 2")] * data_size

    outputs_single_text = [("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label1", 0.21, None),
                           ("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label2", 0.24, None),
                           ("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label3", 0.26, None),
                           ("bfs_conn1", "sub_dir1", "model1",
                            "My test text", "label4", 0.29, None)
                           ] * data_size + \
                          [("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label1", 0.25, None),
                           ("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label2", 0.25, None),
                           ("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label3", 0.25, None),
                           ("bfs_conn1", "sub_dir2", "model1",
                            "My test text", "label4", 0.25, None)] * data_size

    outputs_text_pair = [("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label1", 0.21, None),
                         ("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label2", 0.24, None),
                         ("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label3", 0.26, None),
                         ("bfs_conn1", "sub_dir1", "model1", "My text 1",
                          "My text 2", "label4", 0.29, None)] * data_size + \
                        [("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label1", 0.25, None),
                         ("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label2", 0.25, None),
                         ("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label3", 0.25, None),
                         ("bfs_conn1", "sub_dir2", "model1", "My text 1",
                          "My text 2", "label4", 0.25, None)] * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")
    }

    mock_factory = MockSequenceClassificationFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockSequenceClassificationModel(label_scores=label_scores1),
        PurePosixPath(base_cache_dir1, "sub_dir2", "model1"):
            MockSequenceClassificationModel(label_scores=label_scores2),
    })

    udf_wrapper_single_text = udf_wrapper_single_text
    udf_wrapper_text_pair = udf_wrapper_text_pair
    mock_pipeline = MockPipeline
