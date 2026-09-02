"""Collection of useful bucketfs related operations"""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import (
    Path,
    PurePosixPath,
)
from typing import (
    Literal,
    cast,
)

import exasol.bucketfs as bfs
import fastar
from exasol.saas.client.api_access import get_database_id  # type: ignore

from exasol_transformers_extension.utils.model_specification import ModelSpecification


class ArchiveFormat(Enum):
    """Supported model archive formats."""

    TAR = (".tar", "w")
    TAR_GZ = (".tar.gz", "w:gz")

    @property
    def suffix(self) -> str:
        """Return the filename suffix for this archive format."""
        return self.value[0]

    @property
    def write_mode(self) -> Literal["w", "w:gz"]:
        """Return the fastar write mode for this archive format."""
        return cast(Literal["w", "w:gz"], self.value[1])


def upload_model_files_to_bucketfs(
    model_directory: str,
    bucketfs_model_path: Path,
    bucketfs_location: bfs.path.PathLike,
    archive_format: ArchiveFormat = ArchiveFormat.TAR,
) -> Path:
    """
    uploads model in tmpdir_name to model_path in bucketfs_location
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_upload_tar_file_path = bucketfs_model_path.with_suffix(
            archive_format.suffix
        )
        bucketfs_model_location = bucketfs_location / model_upload_tar_file_path
        archive_path = Path(temporary_directory) / model_upload_tar_file_path.name
        create_tar_of_directory(Path(model_directory), archive_path, archive_format)
        with archive_path.open("rb") as archive_file:
            bucketfs_model_location.write(archive_file)
        return model_upload_tar_file_path


def create_tar_of_directory(
    path: Path,
    archive_path: Path,
    archive_format: ArchiveFormat = ArchiveFormat.TAR,
) -> None:
    """Create a tar archive of "path" at "archive_path"."""
    # fastar.open is provided by the Rust extension and is not visible to Pylint.
    with fastar.open(  # pylint: disable=no-member
        archive_path, archive_format.write_mode
    ) as tar:
        for subpath in path.glob("*"):
            tar.append(path=subpath, arcname=subpath.name)


def get_local_bucketfs_path(
    bucketfs_location: bfs.path.PathLike, model_path: str
) -> PurePosixPath:
    """
    returns path model defined by model_path can be found at in
    bucket defined by bucketfs_location
    """
    bucketfs_model_location = bucketfs_location / model_path
    return PurePosixPath(bucketfs_model_location.as_udf_path())


def create_save_pretrained_model_path(
    _tmpdir_name, model_specification: ModelSpecification
) -> Path:
    """
    path HuggingFaceHubBucketFSModelTransferSP saves the model at using save_pretrained,
    before it is uploaded to the bucketfs
    """
    model_specific_path_suffix = model_specification.get_model_specific_path_suffix()
    return Path(_tmpdir_name, "pretrained", model_specific_path_suffix)


class NotParentError(Exception):
    """
    If the specified PathLike is not a parent of the other.
    """


def relative_to(parent: bfs.path.PathLike, child: bfs.path.PathLike) -> Path:
    prefix = str(parent)
    if not prefix.endswith("/"):
        prefix += "/"
    absolute = str(child)
    if absolute.startswith(prefix):
        return Path(absolute.removeprefix(prefix))
    raise NotParentError(f"{parent} is not a parent of {child}.")
