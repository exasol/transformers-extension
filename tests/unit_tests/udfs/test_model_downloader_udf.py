import pathlib
from tempfile import TemporaryDirectory
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from exasol_transformers_extension.utils import bucketfs_operations
from tests.utils.parameters import model_params


BFS_CONN_NAME = "test_bfs_conn_name"
MODEL_FILE_DATA_MAP = {
    "model_file1.txt": "Sample data in model_file1.txt",
    "model_file2.txt": "Sample data in model_file1.txt"}
TOKENIZER_FILE_DATA_MAP = {
    "tokenizer_file1.txt": "Sample data in tokenizer_file1.txt",
    "tokenizer_file2.txt": "Sample data in tokenizer_file1.txt"}


def udf_wrapper():
    import os
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.models.model_downloader_udf import \
        ModelDownloader

    class MockModelDownloader:
        model_file_data_map = {
            "model_file1.txt": "Sample data in model_file1.txt",
            "model_file2.txt": "Sample data in model_file1.txt"}

        @classmethod
        def from_pretrained(cls, model_name, cache_dir):
            for file_name, content in cls.model_file_data_map.items():
                with open(os.path.join(cache_dir, file_name), 'w') as file:
                    file.write(content)

    class MockTokenizerDownloader:
        tokenizer_file_data_map = {
            "tokenizer_file1.txt": "Sample data in tokenizer_file1.txt",
            "tokenizer_file2.txt": "Sample data in tokenizer_file1.txt"}

        @classmethod
        def from_pretrained(cls, model_name, cache_dir):
            for file_name, content in cls.tokenizer_file_data_map.items():
                with open(os.path.join(cache_dir, file_name), 'w') as file:
                    file.write(content)

    udf = ModelDownloader(exa, base_model_downloader=MockModelDownloader,
                          tokenizer_downloader=MockTokenizerDownloader)

    def run(ctx: UDFContext):
        udf.run(ctx)


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SCALAR",
        input_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bfs_conn", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("outputs", str, "VARCHAR(2000000)")
        ]
    )
    return meta


def test_model_downloader():
    executor = UDFMockExecutor()
    meta = create_mock_metadata()

    with TemporaryDirectory() as path:
        bucketfs_location_read = LocalFSMockBucketFSLocation(path)

        bucketfs_connection = Connection(address=f"file://{path}")
        exa = MockExaEnvironment(
            metadata=meta,
            connections={BFS_CONN_NAME: bucketfs_connection})
        input_data = (
            model_params.name,
            model_params.sub_dir,
            BFS_CONN_NAME)
        result = executor.run([Group([input_data])], exa)

        relative_model_path = str(bucketfs_operations.get_model_path(
            model_params.sub_dir, model_params.name))
        full_model_path = pathlib.PurePath(path, relative_model_path)
        assert result[0].rows[0][0] == relative_model_path \
               and bucketfs_location_read.read_file_from_bucketfs_to_string(
            str(full_model_path.joinpath("model_file1.txt"))) \
               == MODEL_FILE_DATA_MAP["model_file1.txt"] \
               and bucketfs_location_read.read_file_from_bucketfs_to_string(
            str(full_model_path.joinpath("model_file2.txt"))) \
               == MODEL_FILE_DATA_MAP["model_file2.txt"] \
               and bucketfs_location_read.read_file_from_bucketfs_to_string(
            str(full_model_path.joinpath("tokenizer_file1.txt"))) \
               == TOKENIZER_FILE_DATA_MAP["tokenizer_file1.txt"] \
               and bucketfs_location_read.read_file_from_bucketfs_to_string(
            str(full_model_path.joinpath("tokenizer_file2.txt"))) \
               == TOKENIZER_FILE_DATA_MAP["tokenizer_file2.txt"]
