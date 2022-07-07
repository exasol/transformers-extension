from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from exasol_transformers_extension.udfs import bucketfs_operations
from tests.integration_tests.without_db.\
    test_model_downloader_udf_implementation import Connection


MODEL_NAME = "test-model-name"
BFS_CONN_NAME = "test_bfs_conn_name"


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


def test_sequence_classification():
    executor = UDFMockExecutor()
    meta = create_mock_metadata()

    with TemporaryDirectory() as tmpdir_name:
        model_path = PurePosixPath(
            tmpdir_name, bucketfs_operations.get_model_path(MODEL_NAME))
        bucketfs_location = LocalFSMockBucketFSLocation(model_path)

        # upload dummy model files to localfs mock bucketfs
        _upload_dummy_model_files_to_localfs(bucketfs_location, model_path)

        bucketfs_connection = Connection(address=f"file://{model_path}")
        exa = MockExaEnvironment(
            metadata=meta,
            connections={BFS_CONN_NAME: bucketfs_connection})

        input_data = (
            BFS_CONN_NAME,
            str(model_path),  # note that model name is used as model path
            "Test text 1",
        )
        result = executor.run([Group([input_data])], exa)
        logits = result[0].rows
        assert logits == [(0.5, 0.5)]


def _upload_dummy_model_files_to_localfs(bucketfs_location, tmpdir_name):
    model_file_data_map = {
        "model_file1.txt": "Sample data in model_file1.txt",
        "model_file2.txt": "Sample data in model_file1.txt"}
    for file_name, content in model_file_data_map.items():
        bucketfs_location.upload_string_to_bucketfs(
            Path(tmpdir_name, file_name),
            content)
