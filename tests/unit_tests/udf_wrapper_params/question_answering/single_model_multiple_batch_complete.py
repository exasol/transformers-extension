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
        single_model_multiple_batch_complete import \
        SingleModelMultipleBatchComplete as params

    udf = QuestionAnswering(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class SingleModelMultipleBatchComplete:
    """
    single model, multiple batch, last batch complete
    """
    batch_size = 2
    data_size = 4

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1",
                   "question", "context")] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "question",
                    "context", "answer 1", 0.1)] * data_size

    mock_factory = MockQuestionAnsweringFactory({
        "model1": MockQuestionAnsweringModel(answer="answer 1", score=0.1)
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

