import tempfile
from typing import Dict, List
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
    def __init__(self, ctx_data: List[Dict[str, str]]):
        self.ctx_data = ctx_data
        self.index = 0
        self._emitted = []

    @property
    def model_name(self):
        return self.ctx_data[self.index]['base_model']

    @property
    def sub_dir(self):
        return self.ctx_data[self.index]['sub_dir']

    @property
    def bfs_conn(self):
        return self.ctx_data[self.index]['bucketfs_conn_name']

    def next(self):
        self.index += 1
        return None if len(self.ctx_data) == self.index else self.index

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


class TestEnvironmentSetup:
    __test__ = False

    def __init__(self, id: str, url_localfs: str):
        self.bucketfs_conn_name = "bucketfs_connection" + id
        self.sub_dir = model_params.sub_dir + id
        self.base_model = model_params.base_model
        self.ctx_data = {'base_model': self.base_model,
                         'sub_dir': self.sub_dir,
                         'bucketfs_conn_name': self.bucketfs_conn_name}
        self.model_path = bucketfs_operations.get_model_path(
            self.sub_dir, self.base_model)
        self.bucketfs_connection = Connection(
            address=f"{url_localfs}/bucket{id}",
            user=None,
            password=None
        )

    @property
    def bucketfs_location(self):
        return BucketFSFactory().create_bucketfs_location(
            url=self.bucketfs_connection.address,
            user=self.bucketfs_connection.user,
            pwd=self.bucketfs_connection.password)

    def list_files_in_bucketfs(self):
        return self.bucketfs_location.list_files_in_bucketfs(
            str(self.model_path))


def test_model_downloader_udf_implementation():

    with tempfile.TemporaryDirectory() as tmpdir_name:
        url_localfs = f"file://{tmpdir_name}/bucket"
        env1 = TestEnvironmentSetup("1", url_localfs)
        env2 = TestEnvironmentSetup("2", url_localfs)

        ctx = Context([env1.ctx_data, env2.ctx_data])
        exa = ExaEnvironment({
            env1.bucketfs_conn_name: env1.bucketfs_connection,
            env2.bucketfs_conn_name: env2.bucketfs_connection})

        # run udf implementation
        model_downloader = ModelDownloader(exa)
        model_downloader.run(ctx)

        # assertions
        env1_bucketfs_files = env1.list_files_in_bucketfs()
        env2_bucketfs_files = env2.list_files_in_bucketfs()
        assert ctx.get_emitted()[0][0] == str(env1.model_path) \
               and ctx.get_emitted()[1][0] == str(env2.model_path) \
               and env1_bucketfs_files \
               and env2_bucketfs_files
