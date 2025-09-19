from __future__ import annotations

from pathlib import Path
from test.utils.mock_connections import (
    create_mounted_bucketfs_connection,
)
from test.utils.parameters import model_params

import pytest
from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location_from_conn_object,
)
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.delete_models_udf import DeleteModelUDF
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
)
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSPFactory,
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

    def next(self):
        self.index += 1
        return None if len(self.ctx_data) == self.index else self.index

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


class TestEnvironmentSetup:

    def __init__(self, id: str, tmp_dir: Path):
        self.bucketfs_conn_name = "bucketfs_connection" + id
        self.sub_dir = model_params.sub_dir + id
        self.current_model_specs = get_BucketFSModelSpecification_from_model_Specs(
            model_params.tiny_model_specs, self.bucketfs_conn_name, Path(self.sub_dir)
        )
        self.ctx_data = {
            "tiny_model": self.current_model_specs.model_name,
            "task_type": self.current_model_specs.task_type,
            "sub_dir": self.sub_dir,
            "bucketfs_conn_name": self.bucketfs_conn_name,
        }

        self.model_path = self.current_model_specs.get_bucketfs_model_save_path()
        self.bucketfs_connection = create_mounted_bucketfs_connection(
            tmp_dir, f"bucket{id}"
        )

    def upload(self):
        huggingface_hub_bucketfs_model_transfer = (
            HuggingFaceHubBucketFSModelTransferSPFactory()
        )
        with huggingface_hub_bucketfs_model_transfer.create(
            bucketfs_location=self.bucketfs_location,
            model_specification=self.current_model_specs,
            model_path=self.model_path,
            token=None,
        ) as downloader:
            model_factory = self.current_model_specs.get_model_factory()
            downloader.download_from_huggingface_hub(model_factory)
            # upload model files to BucketFS
            return downloader.upload_to_bucketfs()

    @property
    def bucketfs_location(self):
        return create_bucketfs_location_from_conn_object(self.bucketfs_connection)


@pytest.fixture()
def make_test_environment(tmp_path):
    return TestEnvironmentSetup("1", tmp_path)


def test_delete_model_udf_implementation(make_test_environment):
    test_environment = make_test_environment
    model_tar_path = test_environment.upload()

    abs_path = test_environment.bucketfs_location / model_tar_path
    assert abs_path.exists(), abs_path

    ctx = Context([test_environment.ctx_data])
    exa = ExaEnvironment(
        {
            test_environment.bucketfs_conn_name: test_environment.bucketfs_connection,
        }
    )

    # run udf implementation
    delete_model = DeleteModelUDF(exa)
    delete_model.run(ctx)

    assert not abs_path.exists(), abs_path
