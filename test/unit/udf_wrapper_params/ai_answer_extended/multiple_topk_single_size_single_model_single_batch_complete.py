from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_answer_extended.mock_question_answering import (
    MockPipeline,
    MockQuestionAnsweringFactory,
    MockQuestionAnsweringModel,
)

from exasol_udf_mock_python.connection import Connection


def udf_wrapper():
    from test.unit.udf_wrapper_params.ai_answer_extended.mock_sequence_tokenizer import (
        MockSequenceTokenizer,
    )
    from test.unit.udf_wrapper_params.ai_answer_extended.multiple_topk_single_size_single_model_single_batch_complete import (
        MultipleTopkSingleSizeSingleModelNameSingleBatch as params,
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


class MultipleTopkSingleSizeSingleModelNameSingleBatch:
    """
    multiple topk, single size, single model, single batch
    """

    expected_model_counter = 1
    batch_size = 4
    data_size = 2
    top_k1 = 3
    top_k2 = 5

    input_data = [
        (None, "bfs_conn1", "sub_dir1", "model1", "question", "context", top_k1)
    ] * data_size + [
        (None, "bfs_conn1", "sub_dir1", "model1", "question", "context", top_k2)
    ] * data_size
    output_data = [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "question",
            "context",
            top_k1,
            "answer 1",
            0.1,
            1,
            None,
        )
    ] * data_size * top_k1 + [
        (
            "bfs_conn1",
            "sub_dir1",
            "model1",
            "question",
            "context",
            top_k2,
            "answer 1",
            0.1,
            1,
            None,
        )
    ] * data_size * top_k2

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
