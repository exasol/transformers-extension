import tempfile
from typing import Dict
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.udfs.models.model_downloader_udf import \
    ModelDownloader
from tests.utils.parameters import model_params


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
                 sub_dir: str,
                 bfs_conn: str):
        self.model_name = model_name
        self.sub_dir = sub_dir
        self.bfs_conn = bfs_conn
        self._emitted = []

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


def test_model_downloader_udf_implementation():
    bucketfs_conn_name = "bucketfs_connection"
    model_path = bucketfs_operations.get_model_path(
        model_params.sub_dir, model_params.name)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        url_localfs = f"file://{tmpdir_name}/bucket"

        ctx = Context(
            model_params.name, model_params.sub_dir, bucketfs_conn_name)
        bucketfs_connection = Connection(
            address=url_localfs,
            user=None,
            password=None)
        exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

        # run udf implementation
        model_downloader = ModelDownloader(exa)
        model_downloader.run(ctx)

        # assertions
        bucketfs_location = BucketFSFactory().create_bucketfs_location(
            url=bucketfs_connection.address,
            user=bucketfs_connection.user,
            pwd=bucketfs_connection.password)
        bucketfs_files = bucketfs_location.list_files_in_bucketfs(model_path)
        assert ctx.get_emitted()[0][0] == str(model_path) and bucketfs_files
