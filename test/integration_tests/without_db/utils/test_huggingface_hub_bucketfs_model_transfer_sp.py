import tempfile
from pathlib import Path
from test.utils.mock_cast import mock_cast
from test.utils.parameters import model_params
from typing import Union
from unittest.mock import (
    MagicMock,
    Mock,
    create_autospec,
)

import pytest
from transformers import AutoModel

from exasol_transformers_extension.utils.bucketfs_model_uploader import (
    BucketFSModelUploader,
    BucketFSModelUploaderFactory,
)
from exasol_transformers_extension.utils.bucketfs_operations import (
    create_save_pretrained_model_path,
)
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSP,
    ModelFactoryProtocol,
)
from exasol_transformers_extension.utils.temporary_directory_factory import (
    TemporaryDirectoryFactory,
)


class TestSetup:
    __test__ = False

    def __init__(self, bucketfs_location):
        """
        This setup mocks the uploading to BucketFS via
        bucketfs_model_uploader_factory_mock hence avoiding to use an actual
        database.
        """
        self.bucketfs_location = bucketfs_location
        self.temporary_directory_factory = TemporaryDirectoryFactory()
        self.bucketfs_model_uploader_factory_mock: Union[
            BucketFSModelUploaderFactory, MagicMock
        ] = create_autospec(BucketFSModelUploaderFactory)
        self.bucketfs_model_uploader_mock: Union[BucketFSModelUploader, MagicMock] = (
            create_autospec(BucketFSModelUploader)
        )
        mock_cast(self.bucketfs_model_uploader_factory_mock.create).side_effect = [
            self.bucketfs_model_uploader_mock
        ]

        self.token = None
        self.model_specification = model_params.tiny_model_specs
        self.model_path = Path("test_model_path")
        self.downloader = HuggingFaceHubBucketFSModelTransferSP(
            bucketfs_location=self.bucketfs_location,
            bucketfs_model_path=self.model_path,
            model_specification=self.model_specification,
            token=self.token,
            temporary_directory_factory=self.temporary_directory_factory,
            bucketfs_model_uploader_factory=self.bucketfs_model_uploader_factory_mock,
        )

    def reset_mocks(self):
        self.bucketfs_model_uploader_mock.reset_mock()


@pytest.fixture
def bucketfs_location_mock():
    return Mock()


def test_download_with_model(bucketfs_location_mock):
    with tempfile.TemporaryDirectory() as folder:
        test_setup = TestSetup(bucketfs_location_mock)
        base_model_factory: ModelFactoryProtocol = AutoModel
        test_setup.downloader.download_from_huggingface_hub(
            model_factory=base_model_factory
        )
        assert AutoModel.from_pretrained(
            create_save_pretrained_model_path(
                test_setup.downloader._tmpdir_name, test_setup.model_specification
            )
        )
        del test_setup.downloader


def test_download_with_duplicate_model(bucketfs_location_mock):
    with tempfile.TemporaryDirectory() as folder:
        test_setup = TestSetup(bucketfs_location_mock)
        base_model_factory: ModelFactoryProtocol = AutoModel
        test_setup.downloader.download_from_huggingface_hub(
            model_factory=base_model_factory
        )
        test_setup.downloader.download_from_huggingface_hub(
            model_factory=base_model_factory
        )
        assert AutoModel.from_pretrained(
            create_save_pretrained_model_path(
                test_setup.downloader._tmpdir_name, test_setup.model_specification
            )
        )
        del test_setup.downloader
