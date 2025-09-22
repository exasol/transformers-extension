from pathlib import Path
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pytest
from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location_from_conn_object,
)

from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
)
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSPFactory,
)
from exasol_transformers_extension.utils.model_utils import delete_model


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


def test_delete_model(make_test_environment):
    test_env = make_test_environment
    model_tar_path = test_env.upload()

    absolute_path = test_env.bucketfs_location / model_tar_path
    assert absolute_path.exists(), absolute_path

    delete_model(
        bucketfs_location=test_env.bucketfs_location,
        model_spec=test_env.current_model_specs,
    )

    assert not absolute_path.exists(), absolute_path
