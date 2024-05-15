from pathlib import Path

from exasol_bucketfs_utils_python.abstract_bucketfs_location import AbstractBucketFSLocation
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils import bucketfs_operations


class BucketFSModelUploader:

    def __init__(self, bucketfs_model_path: Path, bucketfs_location: BucketFSLocation):
        self._model_path = bucketfs_model_path
        self._bucketfs_location = bucketfs_location

    def upload_directory(self, directory: Path) -> Path:
        return bucketfs_operations.upload_model_files_to_bucketfs(
            model_directory=str(directory),
            bucketfs_model_path=self._model_path,
            bucketfs_location=self._bucketfs_location)


class BucketFSModelUploaderFactory:

    def create(self, model_path: Path, bucketfs_location: BucketFSLocation) -> BucketFSModelUploader:
        return BucketFSModelUploader(bucketfs_model_path=model_path, bucketfs_location=bucketfs_location)
