from pathlib import Path
from typing import Union
from unittest.mock import create_autospec, MagicMock, call

from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils.model_downloader import ModelDownloader, ModelFactoryProtocol
from exasol_transformers_extension.utils.temporary_directory_factory import TemporaryDirectoryFactory
from exasol_transformers_extension.utils.bucketfs_model_uploader import BucketFSModelUploader, \
    BucketFSModelUploaderFactory
from tests.utils.mock_cast import mock_cast


class TestSetup:
    def __init__(self):
        self.bucketfs_location_mock: Union[BucketFSLocation, MagicMock] = create_autospec(BucketFSLocation)
        self.model_factory_mock: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
        self.temporary_directory_factory_mock: Union[TemporaryDirectoryFactory, MagicMock] = \
            create_autospec(TemporaryDirectoryFactory)
        self.bucketfs_model_uploader_factory_mock: Union[BucketFSModelUploaderFactory, MagicMock] = \
            create_autospec(BucketFSModelUploaderFactory)
        self.bucketfs_model_uploader_mock: Union[BucketFSModelUploader, MagicMock] = \
            create_autospec(BucketFSModelUploader)
        mock_cast(self.bucketfs_model_uploader_factory_mock.create).side_effect = [self.bucketfs_model_uploader_mock]

        self.token = "token"
        self.model_name = "test_model_name"
        self.model_path = Path("test_model_path")
        self.downloader = ModelDownloader(
            bucketfs_location=self.bucketfs_location_mock,
            model_path=self.model_path,
            model_name=self.model_name,
            token=self.token,
            temporary_directory_factory=self.temporary_directory_factory_mock,
            bucketfs_model_uploader_factory=self.bucketfs_model_uploader_factory_mock
        )

    def reset_mocks(self):
        self.bucketfs_location_mock.reset_mock()
        self.temporary_directory_factory_mock.reset_mock()
        self.model_factory_mock.reset_mock()
        self.bucketfs_model_uploader_mock.reset_mock()


def test_init():
    test_setup = TestSetup()
    assert test_setup.temporary_directory_factory_mock.mock_calls == [call.create(), call.create().__enter__()] \
           and test_setup.model_factory_mock.mock_calls == [] \
           and test_setup.bucketfs_location_mock.mock_calls == [] \
           and mock_cast(test_setup.bucketfs_model_uploader_factory_mock.create).mock_calls == [
               call.create(model_path=test_setup.model_path, bucketfs_location=test_setup.bucketfs_location_mock)
           ]


def test_download():
    test_setup = TestSetup()
    test_setup.downloader.download_model(model_factory=test_setup.model_factory_mock)
    cache_dir = test_setup.temporary_directory_factory_mock.create().__enter__()
    assert test_setup.model_factory_mock.mock_calls == [
        call.from_pretrained(test_setup.model_name, cache_dir=cache_dir,
                             use_auth_token=test_setup.token)]


def test_upload():
    test_setup = TestSetup()
    test_setup.downloader.download_model(model_factory=test_setup.model_factory_mock)
    test_setup.reset_mocks()
    test_setup.downloader.upload_model()
    cache_dir = test_setup.temporary_directory_factory_mock.create().__enter__()
    assert mock_cast(test_setup.bucketfs_model_uploader_mock.upload_directory).mock_calls == [call(cache_dir)]
