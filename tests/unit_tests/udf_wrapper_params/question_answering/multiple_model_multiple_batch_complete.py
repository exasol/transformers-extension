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
        multiple_model_multiple_batch_complete import \
        MultipleModelMultipleBatchComplete as params

    udf = QuestionAnswering(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class MultipleModelMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete
    """
    batch_size = 2
    data_size = 2

    input_data = \
        [(None, "sub_dir1", "model1", "question", "context")] * data_size + \
        [(None, "sub_dir2", "model2", "question", "context")] * data_size
    output_data = \
        [("sub_dir1", "model1", "question", "context", "answer 1", 0.1)] \
        * data_size + \
        [("sub_dir2", "model2", "question", "context", "answer 2", 0.2)] \
        * data_size

    mock_factory = MockQuestionAnsweringFactory({
        "model1": MockQuestionAnsweringModel(answer="answer 1", score=0.1),
        "model2": MockQuestionAnsweringModel(answer="answer 2", score=0.2)
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

