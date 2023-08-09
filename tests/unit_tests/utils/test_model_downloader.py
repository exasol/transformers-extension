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
        self.bucketfs_location: Union[BucketFSLocation, MagicMock] = create_autospec(BucketFSLocation)
        self.model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
        self.temporary_directory_factory: Union[TemporaryDirectoryFactory, MagicMock] = \
            create_autospec(TemporaryDirectoryFactory)
        self.bucketfs_model_uploader_factory: Union[BucketFSModelUploaderFactory, MagicMock] = \
            create_autospec(BucketFSModelUploaderFactory)
        self.bucketfs_model_uploader: Union[BucketFSModelUploader, MagicMock] = \
            create_autospec(BucketFSModelUploader)
        mock_cast(self.bucketfs_model_uploader_factory.create).side_effect = [self.bucketfs_model_uploader]

        self.token = "token"
        self.model_name = "test_model_name"
        self.model_path = Path("test_model_path")
        self.downloader = ModelDownloader(
            bucketfs_location=self.bucketfs_location,
            model_path=self.model_path,
            model_name=self.model_name,
            token=self.token,
            temporary_directory_factory=self.temporary_directory_factory,
            bucketfs_model_uploader_factory=self.bucketfs_model_uploader_factory
        )


def test_init():
    test_setup = TestSetup()
    assert test_setup.temporary_directory_factory.mock_calls == [] \
           and test_setup.model_factory.mock_calls == [] \
           and test_setup.bucketfs_location.mock_calls == [] \
           and mock_cast(test_setup.bucketfs_model_uploader_factory.create).mock_calls == [
               call.create(model_path=test_setup.model_path, bucketfs_location=test_setup.bucketfs_location)
           ]


def test_download():
    test_setup = TestSetup()
    test_setup.downloader.download_model(model_factory=test_setup.model_factory)
    cache_dir = test_setup.temporary_directory_factory.create().__enter__()
    assert test_setup.model_factory.mock_calls == [
        call.from_pretrained(test_setup.model_name, cache_dir=cache_dir,
                             use_auth_token=test_setup.token)]


def test_upload():
    test_setup = TestSetup()
    test_setup.downloader.download_model(model_factory=test_setup.model_factory)
    cache_dir = test_setup.temporary_directory_factory.create().__enter__()
    assert mock_cast(test_setup.bucketfs_model_uploader.upload_directory).mock_calls == [call(cache_dir)]
