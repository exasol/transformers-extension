from pathlib import Path
from typing import Union
from unittest.mock import create_autospec, MagicMock, call

from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils.bucketfs_model_uploader import BucketFSModelUploader, \
    BucketFSModelUploaderFactory
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import ModelFactoryProtocol, \
    HuggingFaceHubBucketFSModelTransferSP
from exasol_transformers_extension.utils.temporary_directory_factory import TemporaryDirectoryFactory
from tests.utils.mock_cast import mock_cast

from tests.utils.parameters import model_params


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
        model_params_ = model_params.tiny_model
        print(model_params_)
        self.model_name = model_params_
        self.model_path = Path("test_model_path")
        self.downloader = HuggingFaceHubBucketFSModelTransferSP(
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


def test_download_function_call():
    test_setup = TestSetup()
    test_setup.downloader.download_from_huggingface_hub(model_factory=test_setup.model_factory_mock)
    cache_dir = test_setup.temporary_directory_factory_mock.create().__enter__().__truediv__()
    model_save_path = (test_setup.downloader._tmpdir_name/"pretrained"/test_setup.model_name)
    assert test_setup.model_factory_mock.mock_calls == [
        call.from_pretrained(test_setup.model_name, cache_dir=cache_dir,
                             use_auth_token=test_setup.token),
        call.from_pretrained().save_pretrained(model_save_path)]


def test_upload_function_call():
    test_setup = TestSetup()
    test_setup.downloader.download_from_huggingface_hub(model_factory=test_setup.model_factory_mock)
    test_setup.reset_mocks()
    model_save_path = (test_setup.downloader._tmpdir_name/"pretrained"/test_setup.model_name)
    test_setup.downloader.upload_to_bucketfs()
    assert mock_cast(test_setup.bucketfs_model_uploader_mock.upload_directory).mock_calls == [call(model_save_path)]