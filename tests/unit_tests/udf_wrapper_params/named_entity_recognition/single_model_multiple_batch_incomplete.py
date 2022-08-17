from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.named_entity_recognition.\
    mock_named_entity_recognition import \
    MockNamedEntityRecognitionFactory, MockNamedEntityRecognitionModel, MockPipeline


def udf_wrapper():
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.named_entity_recognition_udf import \
        NamedEntityRecognitionUDF
    from tests.unit_tests.udf_wrapper_params.named_entity_recognition. \
        mock_sequence_tokenizer import MockSequenceTokenizer
    from tests.unit_tests.udf_wrapper_params.named_entity_recognition.\
        single_model_multiple_batch_incomplete import \
        SingleModelMultipleBatchIncomplete as params

    udf = NamedEntityRecognitionUDF(
        exa,
        batch_size=params.batch_size,
        pipeline=params.mock_pipeline,
        base_model=params.mock_factory,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


class SingleModelMultipleBatchIncomplete:
    """
    single model, multiple batch, last batch incomplete
    """
    batch_size = 2
    data_size = 5
    n_entities = 3

    input_data = [(None, "bfs_conn1", "sub_dir1", "model1", "text")
                  ] * data_size
    output_data = [("bfs_conn1", "sub_dir1", "model1", "text",
                    1, "text", "label1", 0.1)
                   ] * n_entities * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")}

    mock_factory = MockNamedEntityRecognitionFactory({
        PurePosixPath(base_cache_dir1, "sub_dir1", "model1"):
            MockNamedEntityRecognitionModel(
                indexes=[1] * n_entities,
                words=["text"] * n_entities,
                entities=["label1"] * n_entities,
                scores=[0.1] * n_entities)
    })

    mock_pipeline = MockPipeline
    udf_wrapper = udf_wrapper

