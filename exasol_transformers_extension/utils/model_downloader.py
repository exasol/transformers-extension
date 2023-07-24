import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils import bucketfs_operations


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    def from_pretrained(self, model_name: str, cache_dir: Path, use_auth_token: str):
        pass


class ModelDownloader:

    def __init__(self,
                 bucketfs_location: BucketFSLocation,
                 model_name: str,
                 model_path: Path,
                 token: str):
        self._token = token
        self._model_path = model_path
        self._model_name = model_name
        self._bucketfs_location = bucketfs_location

    def download_model(self, model: ModelFactoryProtocol):
        with tempfile.TemporaryDirectory() as tmpdir_name:
            # download model into tmp folder
            model.from_pretrained(self._model_name, cache_dir=tmpdir_name, use_auth_token=self._token)

            # upload the downloaded model files into bucketfs
            bucketfs_operations.upload_model_files_to_bucketfs(
                tmpdir_name, self._model_path, self._bucketfs_location)


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
