"""Class and factory for uploading a model to the BucketFS."""

from pathlib import Path

import exasol.bucketfs as bfs

from exasol_transformers_extension.utils import bucketfs_operations


class BucketFSModelUploader:
    """Class for uploading a model to the BucketFS."""

    def __init__(self, bucketfs_model_path: Path, bucketfs_location: bfs.path.PathLike):
        """
        :param bucketfs_model_path: Path the model will be put at in the BucketFS.
        :param bucketfs_location: BucketFS location the model will be uploaded to
        """
        self._model_path = bucketfs_model_path
        self._bucketfs_location = bucketfs_location

    def upload_directory(self, directory: Path) -> Path:
        """
        Uploads given (model)-directory to the BucketFS.
        """
        return bucketfs_operations.upload_model_files_to_bucketfs(
            model_directory=str(directory),
            bucketfs_model_path=self._model_path,
            bucketfs_location=self._bucketfs_location,
        )


class BucketFSModelUploaderFactory:
    """Factory for the BucketFSModelUploader class."""

    def create(
        self, model_path: Path, bucketfs_location: bfs.path.PathLike
    ) -> BucketFSModelUploader:
        """
        Returns a BucketFSModelUploader

        :param model_path: Path the model will be put at in the BucketFS.
        :param bucketfs_location: BucketFS location the model should be uploaded to.
        """
        return BucketFSModelUploader(
            bucketfs_model_path=model_path, bucketfs_location=bucketfs_location
        )
