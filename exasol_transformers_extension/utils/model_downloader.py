import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils.bucketfs_model_uploader import BucketFSModelUploaderFactory
from exasol_transformers_extension.utils.temporary_directory_factory import TemporaryDirectoryFactory


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    def from_pretrained(self, model_name: str, cache_dir: Path, use_auth_token: str):
        pass


class ModelDownloader:

    def __init__(self,
                 bucketfs_location: BucketFSLocation,
                 model_name: str,
                 model_path: Path,
                 token: str,
                 temporary_directory_factory: TemporaryDirectoryFactory = TemporaryDirectoryFactory(),
                 bucketfs_model_uploader_factory: BucketFSModelUploaderFactory = BucketFSModelUploaderFactory()):
        self._token = token
        self._model_name = model_name
        self._temporary_directory_factory = temporary_directory_factory
        self._bucketfs_model_uploader = bucketfs_model_uploader_factory.create(
            model_path=model_path,
            bucketfs_location=bucketfs_location)
        self._tmpdir_name = temporary_directory_factory.create().__enter__()

    def __enter__(self):
        return self

    def __del__(self):
        del self._tmpdir_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tmpdir_name.__exit__(exc_type, exc_val, exc_tb)

    def download_model(self, model_factory: ModelFactoryProtocol):
        # download model into tmp folder
        model_factory.from_pretrained(self._model_name, cache_dir=self._tmpdir_name, use_auth_token=self._token)

    def upload_model(self):
        # upload the downloaded model files into bucketfs
        self._bucketfs_model_uploader.upload_directory(self._tmpdir_name)


class ModelDownloaderFactory:

    def create(self,
               bucketfs_location: BucketFSLocation,
               model_name: str,
               model_path: Path,
               token: str) -> ModelDownloader:
        return ModelDownloader(bucketfs_location=bucketfs_location,
                               model_name=model_name,
                               model_path=model_path,
                               token=token)
