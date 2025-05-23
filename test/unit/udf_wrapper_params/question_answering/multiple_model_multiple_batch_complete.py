from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.question_answering.mock_question_answering import (
    MockPipeline,
    MockQuestionAnsweringFactory,
    MockQuestionAnsweringModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.question_answering.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )
    from test.unit.udf_wrapper_params.question_answering.multiple_model_multiple_batch_complete import (
        MultipleModelMultipleBatchComplete as params,
    )

    from exasol_udf_mock_python.udf_context import UDFContext

    from exasol_transformers_extension.udfs.models.question_answering_udf import (
        QuestionAnsweringUDF,
    )

    udf = QuestionAnsweringUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer,
    )

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleModelMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete
    """

    expected_model_counter = 2
    batch_size = 2
    data_size = 2
    top_k = 2

    input_data = [
        (None, "bfs_conn1", "sub_dir1", "model1", "question", "context", top_k)
    ] * data_size + [
        (None, "bfs_conn2", "sub_dir2", "model2", "question", "context", top_k)
    ] * data_size
    output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "question",
            "context",
            top_k,
            "answer 1",
            0.1,
            1,
            None,
        )
    ] * data_size * top_k + [
        (
            "bfs_conn2",
            "sub_dir2",
            "model2",
            "question",
            "context",
            top_k,
            "answer 2",
            0.2,
            1,
            None,
        )
    ] * data_size * top_k

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")

    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
        "bfs_conn2": Connection(address=f"file://{base_cache_dir2}"),
    }
    mock_factory = MockQuestionAnsweringFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 1", score=0.1, rank=1),
            PurePosixPath(
                base_cache_dir2, "sub_dir2", "model2_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 2", score=0.2, rank=1),
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
