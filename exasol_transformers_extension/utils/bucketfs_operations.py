"""Collection of useful bucketfs related operations """
from __future__ import annotations
import tarfile
import tempfile
from pathlib import PurePosixPath, Path
from typing import BinaryIO

import exasol.bucketfs as bfs
from exasol.saas.client.api_access import get_database_id   # type: ignore

from exasol_transformers_extension.utils.model_specification import ModelSpecification


def upload_model_files_to_bucketfs(
        model_directory: str,
        bucketfs_model_path: Path,
        bucketfs_location: bfs.path.PathLike) -> Path:
    """
    uploads model in tmpdir_name to model_path in bucketfs_location
    """
    with tempfile.TemporaryFile() as fileobj:
        create_tar_of_directory(Path(model_directory), fileobj)
        fileobj.seek(0)
        model_upload_tar_file_path = bucketfs_model_path.with_suffix(".tar.gz")
        bucketfs_model_location = bucketfs_location / model_upload_tar_file_path
        bucketfs_model_location.write(fileobj)
        return model_upload_tar_file_path


def create_tar_of_directory(path: Path, fileobj: BinaryIO) -> None:
    """tar the contents of "path" into "fileobj", used for model upload"""
    with tarfile.open(name="model.tar.gz", mode="w|gz", fileobj=fileobj) as tar:
        for subpath in path.glob("*"):
            tar.add(name=subpath, arcname=subpath.name)


def get_local_bucketfs_path(
        bucketfs_location: bfs.path.PathLike, model_path: str) -> PurePosixPath:
    """
    returns path model defined by model_path can be found at in bucket defined by bucketfs_location
    """
    bucketfs_model_location = bucketfs_location / model_path
    return PurePosixPath(bucketfs_model_location.as_udf_path())


def create_save_pretrained_model_path(_tmpdir_name, model_specification: ModelSpecification) -> Path:
    """
    path HuggingFaceHubBucketFSModelTransferSP saves the model at using save_pretrained,
    before it is uploaded to the bucketfs
    """
    model_specific_path_suffix = model_specification.get_model_specific_path_suffix()
    return Path(_tmpdir_name, "pretrained", model_specific_path_suffix)
