import os
from tempfile import TemporaryDirectory
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

MODEL_NAME = "test-model-name"
BFS_CONN_NAME = "test_bfs_conn_name"
FILE_DATA_MAP = {
    "file1.txt": "Sample data in file1.txt",
    "file2.txt": "Sample data in file2.txt"}


def udf_wrapper():
    import os
    from exasol_udf_mock_python.udf_context import UDFContext
    from exasol_transformers_extension.udfs.model_downloader_udf import \
        ModelDownloader

    file_data_map = {
        "file1.txt": "Sample data in file1.txt",
        "file2.txt": "Sample data in file2.txt"}

    class MockModelDownloader:
        @classmethod
        def from_pretrained(cls, model_name, cache_dir):
            for file_name, content in file_data_map.items():
                with open(os.path.join(cache_dir, file_name), 'w') as file:
                    file.write(content)

    udf = ModelDownloader(exa, downloader_method=MockModelDownloader)
    def run(ctx: UDFContext):
        udf.run(ctx)


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SCALAR",
        input_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
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
            MODEL_NAME,
            BFS_CONN_NAME)
        result = executor.run([Group([input_data])], exa)

        relative_model_path = MODEL_NAME.replace('-', '_')
        full_model_path = os.path.join(path, relative_model_path)
        assert result[0].rows[0][0] == relative_model_path \
               and bucketfs_location_read.read_file_from_bucketfs_to_string(
            os.path.join(full_model_path, "file1.txt")) \
               == FILE_DATA_MAP["file1.txt"] \
               and bucketfs_location_read.read_file_from_bucketfs_to_string(
            os.path.join(full_model_path, "file2.txt")) \
               == FILE_DATA_MAP["file2.txt"]
