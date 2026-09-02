from pathlib import Path
from unittest.mock import (
    Mock,
    call,
    patch,
)

import exasol.bucketfs as bfs
import fastar
import pytest
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.utils.bucketfs_model_uploader import (
    BucketFSModelUploader,
)
from exasol_transformers_extension.utils.bucketfs_operations import (
    ArchiveFormat,
    NotParentError,
    create_tar_of_directory,
    relative_to,
    upload_model_files_to_bucketfs,
)


@pytest.fixture
def test_content(tmp_path):
    ref = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
    model_name = Path("test_model_name")
    create_no_exist_directory(model_name, ref, tmp_path)
    create_blobs_directory(model_name, tmp_path)
    create_snapshot_directory(model_name, ref, tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("archive_format", "archive_suffix"),
    [(ArchiveFormat.TAR, ".tar"), (ArchiveFormat.TAR_GZ, ".tar.gz")],
)
def test_upload_model_files_to_bucketfs(
    test_content, tmp_path, archive_format, archive_suffix
):
    path_in_bucket = "abcd"
    bucket = bfs.MountedBucket(base_path=str(tmp_path))
    bucketfs_location = bfs.path.BucketPath(path_in_bucket, bucket)
    model_path = Path("test_model_path")
    with patch(
        "exasol_transformers_extension.utils.bucketfs_operations.tempfile.TemporaryFile",
        side_effect=AssertionError("TemporaryFile must not be used for uploads"),
    ):
        upload_model_files_to_bucketfs(
            bucketfs_location=bucketfs_location,
            bucketfs_model_path=model_path,
            model_directory=str(test_content),
            archive_format=archive_format,
        )
    expected_tar_path = (
        tmp_path / path_in_bucket / model_path.with_suffix(archive_suffix)
    )
    assert expected_tar_path.exists()


def create_no_exist_directory(model_name, ref, tmp_path):
    no_exist = ".no_exist"
    no_exist_path = model_name / no_exist / ref
    (tmp_path / no_exist_path).mkdir(parents=True)
    tokenizer_config_json = "tokenizer_config.json"
    tokenizer_config_json_path = no_exist_path / tokenizer_config_json
    (tmp_path / tokenizer_config_json_path).write_text("tokenizer_config.json")


def create_blobs_directory(model_name, tmp_path):
    blobs = "blobs"
    blobs_path = model_name / blobs
    (tmp_path / blobs_path).mkdir(parents=True)
    blob = "234608c922aaf3989d6a772af31711fbbdd62e3a"
    blob_path = blobs_path / blob
    (tmp_path / blob_path).write_text("blob")


def create_snapshot_directory(model_name, ref, tmp_path):
    snapshots = "snapshots"
    snapshots_path = model_name / snapshots / ref
    (tmp_path / snapshots_path).mkdir(parents=True)
    config_json = "config.json"
    config_path = snapshots_path / config_json
    (tmp_path / config_path).write_text("config.json")


@pytest.mark.parametrize(
    ("archive_format", "archive_suffix"),
    [(ArchiveFormat.TAR, ".tar"), (ArchiveFormat.TAR_GZ, ".tar.gz")],
)
def test_create_tar_of_directory(
    test_content, tmp_path, archive_format, archive_suffix
):
    archive_path = tmp_path.parent / f"{tmp_path.name}{archive_suffix}"
    create_tar_of_directory(test_content, archive_path, archive_format)
    extracted_archive_path = tmp_path / "extracted"
    with fastar.open(archive_path, "r") as archive:
        archive.unpack(extracted_archive_path)

    extracted_model_path = extracted_archive_path / "test_model_name"
    assert extracted_model_path.is_dir()
    assert sorted(
        str(path.relative_to(extracted_archive_path))
        for path in extracted_archive_path.rglob("*")
    ) == [
        "test_model_name",
        "test_model_name/.no_exist",
        "test_model_name/.no_exist/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837",
        "test_model_name/.no_exist/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/tokenizer_config.json",
        "test_model_name/blobs",
        "test_model_name/blobs/234608c922aaf3989d6a772af31711fbbdd62e3a",
        "test_model_name/snapshots",
        "test_model_name/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837",
        "test_model_name/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/config.json",
    ]


@patch(
    "exasol_transformers_extension.utils.bucketfs_model_uploader.bucketfs_operations"
)
def test_model_uploader_forwards_archive_format(bucketfs_operations_mock, tmp_path):
    uploader = BucketFSModelUploader(Path("model"), Mock())

    uploader.upload_directory(tmp_path, ArchiveFormat.TAR_GZ)

    assert bucketfs_operations_mock.upload_model_files_to_bucketfs.call_args == call(
        model_directory=str(tmp_path),
        bucketfs_model_path=Path("model"),
        bucketfs_location=uploader._bucketfs_location,
        archive_format=ArchiveFormat.TAR_GZ,
    )


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("a/b", "a/b/c", "c"),
        ("a/b", "a/b/c/d", "c/d"),
        ("/", "/a/b", "a/b"),
        ("/a/b", "/a/b/c", "c"),
        ("/a/b/", "/a/b/c/", "c"),
    ],
)
def test_relative_to(a, b, expected):
    parent = bfs._path.BucketPath(a, Mock())
    child = bfs._path.BucketPath(b, Mock())
    assert relative_to(parent, child) == Path(expected)


@pytest.mark.parametrize(
    "a, b",
    [
        ("a/b", "d"),
        ("a/b", "/d"),
        ("a/b", "a/c"),
        ("/a/b", "/a/c"),
        ("/a/b/", "/a/c/"),
        ("a/b/", "a/c/"),
    ],
)
def test_not_relative_to(a, b):
    parent = bfs._path.BucketPath(a, Mock())
    child = bfs._path.BucketPath(b, Mock())
    with pytest.raises(NotParentError):
        relative_to(parent, child)
