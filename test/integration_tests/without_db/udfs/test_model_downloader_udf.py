from __future__ import annotations

from pathlib import Path
from test.utils.bucketfs_file_list import get_bucketfs_file_list
from test.utils.mock_connections import (
    create_hf_token_connection,
    create_mounted_bucketfs_connection,
)
from test.utils.parameters import model_params
from typing import (
    Dict,
    List,
)

from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location_from_conn_object,
)
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.model_downloader_udf import (
    ModelDownloaderUDF,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
)


class ExaEnvironment:
    def __init__(self, connections: dict[str, Connection] = None):
        self._connections = connections
        if self._connections is None:
            self._connections = {}

    def get_connection(self, name: str) -> Connection:
        return self._connections[name]


class Context:
    def __init__(self, ctx_data: list[dict[str, str]]):
        self.ctx_data = ctx_data
        self.index = 0
        self._emitted = []

    @property
    def model_name(self):
        return self.ctx_data[self.index]["tiny_model"]

    @property
    def sub_dir(self):
        return self.ctx_data[self.index]["sub_dir"]

    @property
    def task_type(self):
        return self.ctx_data[self.index]["task_type"]

    @property
    def bfs_conn(self):
        return self.ctx_data[self.index]["bucketfs_conn_name"]

    @property
    def token_conn(self):
        return self.ctx_data[self.index]["token_conn_name"]

    def next(self):
        self.index += 1
        return None if len(self.ctx_data) == self.index else self.index

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


class TestEnvironmentSetup:
    __test__ = False

    def __init__(self, id: str, tmp_dir: Path, token_conn_name: str):
        self.bucketfs_conn_name = "bucketfs_connection" + id
        self.sub_dir = model_params.sub_dir + id
        current_model_specs = get_BucketFSModelSpecification_from_model_Specs(
            model_params.tiny_model_specs, self.bucketfs_conn_name, Path(self.sub_dir)
        )
        self.token_conn_name = token_conn_name
        self.ctx_data = {
            "tiny_model": current_model_specs.model_name,
            "task_type": current_model_specs.task_type,
            "sub_dir": self.sub_dir,
            "bucketfs_conn_name": self.bucketfs_conn_name,
            "token_conn_name": self.token_conn_name,
        }

        self.model_path = current_model_specs.get_bucketfs_model_save_path()
        self.bucketfs_connection = create_mounted_bucketfs_connection(
            tmp_dir, f"bucket{id}"
        )
        self.token_connection = (
            None if not self.token_conn_name else create_hf_token_connection("valid")
        )

    def list_files_in_bucketfs(self):
        bucketfs_location = create_bucketfs_location_from_conn_object(
            self.bucketfs_connection
        )
        return get_bucketfs_file_list(bucketfs_location)


def test_model_downloader_udf_implementation(tmp_path):
    env1 = TestEnvironmentSetup("1", tmp_path, token_conn_name="")

    ctx = Context([env1.ctx_data])
    exa = ExaEnvironment(
        {
            env1.bucketfs_conn_name: env1.bucketfs_connection,
        }
    )

    # run udf implementation
    model_downloader = ModelDownloaderUDF(exa)
    model_downloader.run(ctx)

    # assertions
    env1_bucketfs_files = env1.list_files_in_bucketfs()
    assert ctx.get_emitted()[0] == (
        str(env1.model_path),
        str(env1.model_path.with_suffix(".tar.gz")),
    )
    assert ctx.get_emitted()[0][1] in env1_bucketfs_files
