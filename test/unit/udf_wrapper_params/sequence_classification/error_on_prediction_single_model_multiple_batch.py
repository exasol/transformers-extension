from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.sequence_classification.mock_sequence_classification_factory import (
    LabelScore,
    MockPipeline,
    MockSequenceClassificationFactory,
    MockSequenceClassificationModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper_single_text():
    from test.unit.udf_wrapper_params.sequence_classification.error_on_prediction_single_model_multiple_batch import (
        ErrorOnPredictionSingleModelMultipleBatch as params,
    )
    from test.unit.udf_wrapper_params.sequence_classification.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.sequence_classification_single_text_udf import (
        SequenceClassificationSingleTextUDF,
    )

    udf = SequenceClassificationSingleTextUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


def udf_wrapper_text_pair():
    from test.unit.udf_wrapper_params.sequence_classification.error_on_prediction_single_model_multiple_batch import (
        ErrorOnPredictionSingleModelMultipleBatch as params,
    )
    from test.unit.udf_wrapper_params.sequence_classification.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.sequence_classification_text_pair_udf import (
        SequenceClassificationTextPairUDF,
    )

    udf = SequenceClassificationTextPairUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


class ErrorOnPredictionSingleModelMultipleBatch:
    """
    error on prediction, single model, multiple batch,
    """

    expected_single_text_model_counter = 1
    expected_text_pair_model_counter = 1
    batch_size = 2
    data_size = 5

    label_scores = [
        LabelScore("label1", 0.21),
        LabelScore("label2", 0.24),
        LabelScore("label3", 0.26),
        LabelScore("label4", 0.29),
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockSequenceClassificationFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_text-classification"
            ): MockSequenceClassificationModel(label_scores=label_scores),
        }
    )

    inputs_single_text = [
        (None, "bfs_conn1", "sub_dir1", "model1", "error on pred")
    ] * data_size
    inputs_pair_text = [
        (None, "bfs_conn1", "sub_dir1", "model1", "error on pred", "My text 2")
    ] * data_size

    outputs_single_text = [
        ("bfs_conn1", "sub_dir1", "model1", "error on pred", None, None, "Traceback"),
        ("bfs_conn1", "sub_dir1", "model1", "error on pred", None, None, "Traceback"),
        ("bfs_conn1", "sub_dir1", "model1", "error on pred", None, None, "Traceback"),
        ("bfs_conn1", "sub_dir1", "model1", "error on pred", None, None, "Traceback"),
    ] * data_size
    outputs_text_pair = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "error on pred",
            "My text 2",
            None,
            None,
            "Traceback",
        ),
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "error on pred",
            "My text 2",
            None,
            None,
            "Traceback",
        ),
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "error on pred",
            "My text 2",
            None,
            None,
            "Traceback",
        ),
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "error on pred",
            "My text 2",
            None,
            None,
            "Traceback",
        ),
    ] * data_size

    udf_wrapper_single_text = udf_wrapper_single_text
    udf_wrapper_text_pair = udf_wrapper_text_pair
    mock_pipeline = MockPipeline
