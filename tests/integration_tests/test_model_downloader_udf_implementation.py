from pathlib import Path
from typing import Dict
from exasol_bucketfs_utils_python import list_files, delete
from exasol_transformers_extension.udfs.model_downloader_udf import \
    ModelDownloader
from tests.utils.parameters import bucketfs_params


class Connection:
    def __init__(self, address: str, user: str = None, password: str = None):
        self.address = address
        self.user = user
        self.password = password


class ExaEnvironment:
    def __init__(self, connections: Dict[str, Connection] = None):
        self._connections = connections
        if self._connections is None:
            self._connections = {}

    def get_connection(self, name: str) -> Connection:
        return self._connections[name]


class Context:
    def __init__(self,
                 model_name: str,
                 bfs_conn: str):
        self.model_name = model_name
        self.bfs_conn = bfs_conn
        self._emitted = []

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


def test_model_downloader_udf_implementation(setup_database, bucket_config):
    bucketfs_conn_name = "bucketfs_connection"
    model_name = 'bert-base-uncased'
    model_path = model_name.replace("-", "_")

    ctx = Context(
        model_name,
        bucketfs_conn_name
    )

    bucketfs_connection = Connection(
        address=bucketfs_params.address(),
        user=bucketfs_params.user,
        password=bucketfs_params.password)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    try:
        model_downloader = ModelDownloader(exa)
        model_downloader .run(ctx)

        # assertions
        path_in_the_bucket = str(Path("container", f"{model_path}"))
        files = list_files.list_files_in_bucketfs(
            bucket_config, path_in_the_bucket)
        assert ctx.get_emitted()[0][0] == model_path and files
    finally:
        # revert, delete downloaded model files
        try:
            files = list_files.list_files_in_bucketfs(
                bucket_config, path_in_the_bucket)
            for file_ in files:
                delete.delete_file_in_bucketfs(
                    bucket_config, str(Path(path_in_the_bucket, file_)))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")
