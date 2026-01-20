from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_answer_extended.mock_question_answering import (
    MockPipeline,
    MockQuestionAnsweringFactory,
    MockQuestionAnsweringModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.ai_answer_extended.error_on_prediction_single_model_multiple_batch import (
        ErrorOnPredictionSingleModelMultipleBatch as params,
    )
    from test.unit.udf_wrapper_params.ai_answer_extended.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.ai_answer_extended_udf import (
        AiAnswerExtendedUDF,
    )

    udf = AiAnswerExtendedUDF(
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

    expected_model_counter = 1
    batch_size = 2
    data_size = 4
    top_k = 2

    input_data = [
        (None, "bfs_conn1", "sub_dir1", "model1", "question", "error on pred", top_k)
    ] * data_size
    output_data = (
        [
            (
                "bfs_conn1",
                "sub_dir1",
                "model1",
                "question",
                "error on pred",
                top_k,
                None,
                None,
                None,
                "Traceback",
            )
        ]
        * data_size
        * top_k
    )

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockQuestionAnsweringFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 1", score=0.1, rank=1)
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
