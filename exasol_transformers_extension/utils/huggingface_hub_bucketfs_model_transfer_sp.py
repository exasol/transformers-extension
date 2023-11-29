import os
import tempfile
from pathlib import Path
from typing import Protocol, Union, runtime_checkable

import transformers
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils.bucketfs_model_uploader import BucketFSModelUploaderFactory
from exasol_transformers_extension.utils.temporary_directory_factory import TemporaryDirectoryFactory


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    def from_pretrained(self, model_name: str, cache_dir: Path, use_auth_token: str) -> transformers.PreTrainedModel:
        pass

    def save_pretrained(self, save_directory: Union[str, Path]):
        pass


class HuggingFaceHubBucketFSModelTransferSP:
    def __init__(self,
                 bucketfs_location: BucketFSLocation,
                 model_name: str,
                 model_path: Path,
                 local_model_save_path: Path,
                 token: str,
                 temporary_directory_factory: TemporaryDirectoryFactory = TemporaryDirectoryFactory(),
                 bucketfs_model_uploader_factory: BucketFSModelUploaderFactory = BucketFSModelUploaderFactory()):
        self._token = token
        self._model_name = model_name
        self._local_model_save_path = Path(local_model_save_path)
        self._temporary_directory_factory = temporary_directory_factory
        self._bucketfs_model_uploader = bucketfs_model_uploader_factory.create(
            model_path=model_path,
            bucketfs_location=bucketfs_location)
        self._tmpdir = temporary_directory_factory.create()
        self._tmpdir_name = self._tmpdir.__enter__()

    def __enter__(self):
        return self

    def __del__(self):
        self._tmpdir.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tmpdir.__exit__(exc_type, exc_val, exc_tb)

    def download_from_huggingface_hub_sp(self, model_factory: ModelFactoryProtocol):
        """
        Download a model from HuggingFace Hub into a temporary directory and save it with save_pretrained
        at _local_model_save_path / _model_name for local storing
        """
        model = model_factory.from_pretrained(self._model_name, cache_dir=self._tmpdir_name, use_auth_token=self._token)
        path = self._local_model_save_path / self._model_name
        model.save_pretrained(path) #todo save in cachedir in assuption will be uploaded and then deleted?

    def upload_to_bucketfs(self) -> Path:
        """
        Upload the downloaded models into the BucketFS
        """
        return self._bucketfs_model_uploader.upload_directory(self._tmpdir_name)


class HuggingFaceHubBucketFSModelTransferSPFactory:

    def create(self,
               bucketfs_location: BucketFSLocation,
               model_name: str,
               model_path: Path,
               local_model_save_path: Path,
               token: str) -> HuggingFaceHubBucketFSModelTransferSP:
        return HuggingFaceHubBucketFSModelTransferSP(bucketfs_location=bucketfs_location,
                                                     model_name=model_name,
                                                     model_path=model_path,
                                                     local_model_save_path=local_model_save_path,
                                                     token=token)
