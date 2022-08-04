import tempfile
from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.question_answering.\
    mock_question_answering import \
    MockQuestionAnsweringFactory, MockQuestionAnsweringModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.question_answering_udf import \
        QuestionAnswering
    from tests.unit_tests.udf_wrapper_params.question_answering. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.question_answering.\
        multiple_model_multiple_batch_multiple_models_per_batch import \
        MultipleModelMultipleBatchMultipleModelsPerBatch as params

    udf = QuestionAnswering(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleModelMultipleBatchMultipleModelsPerBatch:
    """
    multiple model, multiple batch, multiple models per batch
    """
    batch_size = 2
    data_size = 1

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "question", "context")] * data_size + \
                 [(None, "bfs_conn2", "sub_dir2", "model2",
                   "question", "context")] * data_size + \
                 [(None, "bfs_conn3", "sub_dir3", "model3",
                   "question", "context")] * data_size + \
                 [(None, "bfs_conn4", "sub_dir4", "model4",
                   "question", "context")] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "question",
                    "context", "answer 1", 0.1)] * data_size + \
                  [("bfs_conn2", "sub_dir2", "model2", "question",
                    "context", "answer 2", 0.2)] * data_size + \
                  [("bfs_conn3", "sub_dir3", "model3", "question",
                    "context", "answer 3", 0.3)] * data_size + \
                  [("bfs_conn4", "sub_dir4", "model4", "question",
                    "context", "answer 4", 0.4)] * data_size

    with tempfile.TemporaryDirectory() as tmpdir_name:
        base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
        base_cache_dir2 = PurePosixPath(tmpdir_name, "bfs_conn2")
        cache_dir3 = PurePosixPath(tmpdir_name, "bfs_conn3")
        cache_dir4 = PurePosixPath(tmpdir_name, "bfs_conn4")

        bfs_connections = {
            "bfs_conn1": Connection(address=f"file://{base_cache_dir1}"),
            "bfs_conn2": Connection(address=f"file://{base_cache_dir2}"),
            "bfs_conn3": Connection(address=f"file://{cache_dir3}"),
            "bfs_conn4": Connection(address=f"file://{cache_dir4}")
        }

        mock_factory = MockQuestionAnsweringFactory({
            PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
                MockQuestionAnsweringModel(answer="answer 1", score=0.1),
            PurePosixPath(base_cache_dir2, "sub_dir2", "model2"):
                MockQuestionAnsweringModel(answer="answer 2", score=0.2),
            PurePosixPath(cache_dir3, "sub_dir3", "model3"):
                MockQuestionAnsweringModel(answer="answer 3", score=0.3),
            PurePosixPath(cache_dir4, "sub_dir4", "model4"):
                MockQuestionAnsweringModel(answer="answer 4", score=0.4),
        })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

