import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest
import exasol.bucketfs as bfs
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.utils.bucketfs_operations import (
    create_bucketfs_location,
    create_bucketfs_location_from_conn_object,
    upload_model_files_to_bucketfs,
    create_tar_of_directory
)


@pytest.fixture
def test_content(tmp_path):
    ref = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
    model_name = Path("test_model_name")
    create_no_exist_directory(model_name, ref, tmp_path)
    create_blobs_directory(model_name, tmp_path)
    create_snapshot_directory(model_name, ref, tmp_path)
    return tmp_path


@patch("exasol.bucketfs.path.build_path")
def test_create_bucketfs_location_from_conn_object(mock_build_path):
    url = 'https://bucket-fs-service'
    bucket = 'my-bucket'
    user = 'the-user'
    password = 'the-password'
    conn = Connection(
        address=f'{{"url":"{url}", "bucket":"{bucket}"}}',
        user=f'{{"user":"{user}"}}',
        password=f'{{"password":"{password}"}}'
    )
    create_bucketfs_location_from_conn_object(conn)
    mock_build_path.assert_called_with(url=url, bucket=bucket, user=user, password=password)


@patch("exasol.bucketfs.path.build_path")
def test_create_bucketfs_location_on_prem(mock_build_path):
    create_bucketfs_location(bucketfs_host='https://bucket-fs-service', bucketfs_port=5678,
                             bucketfs_name='bfs-service', bucket='my-bucket', bucketfs_user='bfs-user',
                             bucketfs_password='bfs-password', path_in_bucket='bucket_path')
    assert mock_build_path.call_args.kwargs['backend'] == bfs.path.StorageBackend.onprem


@patch("exasol.bucketfs.path.build_path")
def test_create_bucketfs_location_saas(mock_build_path):
    create_bucketfs_location(saas_url='https://saas-service', saas_account_id='fake-account-id',
                             saas_database_id='fake-database-id', saas_token='fake-saas-token',
                             path_in_bucket='bucket_path')
    assert mock_build_path.call_args.kwargs['backend'] == bfs.path.StorageBackend.saas


def test_upload_model_files_to_bucketfs(test_content, tmp_path):
    path_in_backet = 'abcd'
    bucket = bfs.MountedBucket(base_path=str(tmp_path))
    bucketfs_location = bfs.path.BucketPath(path_in_backet, bucket)
    model_path = Path("test_model_path")
    upload_model_files_to_bucketfs(
        bucketfs_location=bucketfs_location,
        bucketfs_model_path=model_path,
        model_directory=str(test_content)
    )
    expected_tar_path = tmp_path / path_in_backet / model_path.with_suffix(".tar.gz")
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


def test_create_tar_of_directory(test_content):
    fileobj = io.BytesIO()
    create_tar_of_directory(test_content, fileobj)
    fileobj.seek(0)
    with tarfile.open(name="test.tar.gz", mode="r|gz", fileobj=fileobj) as tar:
        assert tar.getnames() == [
            'test_model_name',
            'test_model_name/.no_exist',
            'test_model_name/.no_exist/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837',
            'test_model_name/.no_exist/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/tokenizer_config.json',
            'test_model_name/blobs',
            'test_model_name/blobs/234608c922aaf3989d6a772af31711fbbdd62e3a',
            'test_model_name/snapshots',
            'test_model_name/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837',
            'test_model_name/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837/config.json']
