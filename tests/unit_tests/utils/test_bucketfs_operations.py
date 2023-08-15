import unittest.mock
from pathlib import Path
from typing import Union
from unittest.mock import create_autospec, MagicMock, call

from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils.bucketfs_operations import upload_model_files_to_bucketfs
from tests.utils.matchers import AnyOrder


def test_upload_model_files_to_bucketfs(tmp_path):
    mock_bucketfs_location: Union[BucketFSLocation, MagicMock] = create_autospec(BucketFSLocation)
    model_path = Path("test_model_path")
    ref = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
    model_name = Path("test_model_name")
    create_no_exits_directory(model_name, ref, tmp_path)
    create_blobs_directory(model_name, tmp_path)
    create_snapshot_directory(model_name, ref, tmp_path)
    upload_model_files_to_bucketfs(
        bucketfs_location=mock_bucketfs_location,
        model_path=model_path,
        tmpdir_name=str(tmp_path)
    )
    assert mock_bucketfs_location.mock_calls == AnyOrder([
        call.upload_string_to_bucketfs('test_model_path/test_model_name/.not_empty', ''),
        call.upload_string_to_bucketfs('test_model_path/test_model_name/snapshots/.not_empty', ''),
        call.upload_string_to_bucketfs('test_model_path/test_model_name/blobs/.not_empty', ''),
        call.upload_string_to_bucketfs('test_model_path/test_model_name/.no_exist/.not_empty', ''),
        call.upload_string_to_bucketfs(
            'test_model_path/test_model_name/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/.not_empty', ''),
        call.upload_fileobj_to_bucketfs(
            unittest.mock.ANY,
            'test_model_path/test_model_name/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/config.json'),
        call.upload_fileobj_to_bucketfs(
            unittest.mock.ANY,
            'test_model_path/test_model_name/blobs/234608c922aaf3989d6a772af31711fbbdd62e3a'),
        call.upload_string_to_bucketfs(
            'test_model_path/test_model_name/.no_exist/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/.not_empty', ''),
        call.upload_fileobj_to_bucketfs(
            unittest.mock.ANY,
            'test_model_path/test_model_name/.no_exist/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/tokenizer_config.json')
    ])


def create_no_exits_directory(model_name, ref, tmp_path):
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
