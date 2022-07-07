from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor


def udf_wrapper():
    import torch
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.\
        sequence_classification_single_text_udf import \
        SequenceClassificationSingleText

    class MockSequenceClassification:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, model_name, cache_dir):
            return cls

        @property
        def logits(self) -> torch.FloatTensor:
            return torch.FloatTensor([[0.1, 0.1]])

    class MockSequenceTokenizer:
        def __new__(cls, text: str, return_tensors: str):
            return {}

        @classmethod
        def from_pretrained(cls, model_name, cache_dir):
            return cls

    udf = SequenceClassificationSingleText(
        exa,
        cache_dir="dummy_cache_dir",
        base_model=MockSequenceClassification,
        tokenizer=MockSequenceTokenizer)

    def run(ctx: UDFContext):
        udf.run(ctx)


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("label_0", float, "DOUBLE"),
            Column("label_1", float, "DOUBLE"),
        ],
    )
    return meta


def test_sequence_classification_single_text(
        upload_dummy_model_to_local_bucketfs):

    bucketfs_connection_name = "test_bfs_conn_name"
    model_path = upload_dummy_model_to_local_bucketfs

    executor = UDFMockExecutor()
    meta = create_mock_metadata()
    bucketfs_connection = Connection(address=f"file://{model_path}")
    exa = MockExaEnvironment(
        metadata=meta,
        connections={bucketfs_connection_name: bucketfs_connection})

    input_data = (
        bucketfs_connection_name,
        str(model_path),
        "Test text 1")
    result = executor.run([Group([input_data])], exa)

    logits = result[0].rows
    assert logits == [(0.5, 0.5)]
