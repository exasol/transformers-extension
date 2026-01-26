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
    from test.unit.udf_wrapper_params.ai_answer_extended.multiple_topk_multiple_size_single_model_multiple_batch_complete import (
        MultipleTopkMultipleSizeSingleModelNameMultipleBatch as params,
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


class MultipleTopkMultipleSizeSingleModelNameMultipleBatch:
    """
    multiple topk, multiple size, single model, multiple batch
    """

    expected_model_counter = 4
    batch_size = 2
    data_size1 = 1
    data_size2 = 2
    top_k1_for_datasize1 = 1
    top_k2_for_datasize1 = 2
    top_k3_for_datasize2 = 1
    top_k4_for_datasize2 = 2

    input_data = (
        [
            (
                None,
                "bfs_conn1",
                "sub_dir1",
                "model1",
                "question",
                "context",
                top_k1_for_datasize1,
            )
        ]
        * data_size1
        + [
            (
                None,
                "bfs_conn1",
                "sub_dir1",
                "model2",
                "question",
                "context",
                top_k2_for_datasize1,
            )
        ]
        * data_size1
        + [
            (
                None,
                "bfs_conn1",
                "sub_dir1",
                "model3",
                "question",
                "context",
                top_k3_for_datasize2,
            )
        ]
        * data_size2
        + [
            (
                None,
                "bfs_conn1",
                "sub_dir1",
                "model4",
                "question",
                "context",
                top_k4_for_datasize2,
            )
        ]
        * data_size2
    )

    output_data = (
        [
            (
                "bfs_conn1",
                "sub_dir1",
                "model1",
                "question",
                "context",
                top_k1_for_datasize1,
                "answer 1",
                0.1,
                1,
                None,
            )
        ]
        * data_size1
        * top_k1_for_datasize1
        + [
            (
                "bfs_conn1",
                "sub_dir1",
                "model2",
                "question",
                "context",
                top_k2_for_datasize1,
                "answer 2",
                0.2,
                1,
                None,
            )
        ]
        * data_size1
        * top_k2_for_datasize1
        + [
            (
                "bfs_conn1",
                "sub_dir1",
                "model3",
                "question",
                "context",
                top_k3_for_datasize2,
                "answer 3",
                0.3,
                1,
                None,
            )
        ]
        * data_size2
        * top_k3_for_datasize2
        + [
            (
                "bfs_conn1",
                "sub_dir1",
                "model4",
                "question",
                "context",
                top_k4_for_datasize2,
                "answer 4",
                0.4,
                1,
                None,
            )
        ]
        * data_size2
        * top_k4_for_datasize2
    )

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {"bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockQuestionAnsweringFactory(
        {
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model1_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 1", score=0.1, rank=1),
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model2_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 2", score=0.2, rank=1),
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model3_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 3", score=0.3, rank=1),
            PurePosixPath(
                base_cache_dir1, "sub_dir1", "model4_question-answering"
            ): MockQuestionAnsweringModel(answer="answer 4", score=0.4, rank=1),
        }
    )

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper
