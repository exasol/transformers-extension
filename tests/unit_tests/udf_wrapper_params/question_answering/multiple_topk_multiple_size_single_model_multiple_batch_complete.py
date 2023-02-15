from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.question_answering.\
    mock_question_answering import \
    MockQuestionAnsweringFactory, MockQuestionAnsweringModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.question_answering_udf import \
        QuestionAnsweringUDF
    from tests.unit_tests.udf_wrapper_params.question_answering. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.question_answering.\
        multiple_topk_multiple_size_single_model_multiple_batch_complete import \
        MultipleTopkMultipleSizeSingleModelNameMultipleBatch as params

    udf = QuestionAnsweringUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleTopkMultipleSizeSingleModelNameMultipleBatch:
    """
    multiple topk, multiple size, single model, multiple batch
    """
    batch_size = 2
    data_size1 = 1
    data_size2 = 2
    top_k1_for_datasize1 = 1
    top_k2_for_datasize1 = 2
    top_k3_for_datasize2 = 1
    top_k4_for_datasize2 = 2

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1", "question",
                   "context", top_k1_for_datasize1)] * data_size1 + \
                 [(None, "bfs_conn1", "sub_dir1", "model2", "question",
                   "context", top_k2_for_datasize1)] * data_size1 + \
                 [(None, "bfs_conn1", "sub_dir1", "model3", "question",
                   "context", top_k3_for_datasize2)] * data_size2 + \
                 [(None, "bfs_conn1", "sub_dir1", "model4", "question",
                   "context", top_k4_for_datasize2)] * data_size2

    output_data = [("bfs_conn1", "sub_dir1", "model1", "question", "context",
                    top_k1_for_datasize1, "answer 1", 0.1, 1)
                   ] * data_size1 * top_k1_for_datasize1 + \
                  [("bfs_conn1", "sub_dir1", "model2", "question", "context",
                    top_k2_for_datasize1, "answer 2", 0.2, 1)
                   ] * data_size1 * top_k2_for_datasize1 + \
                  [("bfs_conn1", "sub_dir1", "model3", "question", "context",
                    top_k3_for_datasize2, "answer 3", 0.3, 1)
                   ] * data_size2 * top_k3_for_datasize2 + \
                  [("bfs_conn1", "sub_dir1", "model4", "question", "context",
                    top_k4_for_datasize2, "answer 4", 0.4, 1)
                   ] * data_size2 * top_k4_for_datasize2

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockQuestionAnsweringFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockQuestionAnsweringModel(answer="answer 1", score=0.1, rank=1),
        PurePosixPath(base_cache_dir1, "sub_dir1", "model2"):
            MockQuestionAnsweringModel(answer="answer 2", score=0.2, rank=1),
        PurePosixPath(base_cache_dir1, "sub_dir1", "model3"):
            MockQuestionAnsweringModel(answer="answer 3", score=0.3, rank=1),
        PurePosixPath(base_cache_dir1, "sub_dir1", "model4"):
            MockQuestionAnsweringModel(answer="answer 4", score=0.4, rank=1),
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

