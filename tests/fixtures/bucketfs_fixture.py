from urllib.parse import urlparse

import pytest
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from pytest_itde.config import TestConfig
import exasol.bucketfs as bfs

from exasol_transformers_extension.utils.bucketfs_operations import create_bucketfs_location
from tests.utils.parameters import bucketfs_params
from exasol_bucketfs_utils_python.bucket_config import BucketConfig
from exasol_bucketfs_utils_python.bucketfs_config import BucketFSConfig
from exasol_bucketfs_utils_python.bucketfs_connection_config import \
    BucketFSConnectionConfig


@pytest.fixture(scope="session")
def bucket_config(itde: TestConfig) -> BucketConfig:
    parsed_url = urlparse(itde.bucketfs.url)
    connection_config = BucketFSConnectionConfig(
        host=parsed_url.hostname, port=parsed_url.port,
        user=itde.bucketfs.username, pwd=itde.bucketfs.password,
        is_https=False)
    bucketfs_config = BucketFSConfig(
        connection_config=connection_config, bucketfs_name=bucketfs_params.name)
    bucket_config = BucketConfig(
        bucket_name=bucketfs_params.bucket, bucketfs_config=bucketfs_config)
    return bucket_config


@pytest.fixture(scope="session")
def bucketfs_location(itde: TestConfig) -> bfs.path.PathLike:
    return create_bucketfs_location(
        path_in_bucket=bucketfs_params.path_in_bucket,
        bucketfs_name=bucketfs_params.name,
        bucketfs_url=itde.bucketfs.url,
        bucketfs_user=itde.bucketfs.username,
        bucketfs_password=itde.bucketfs.password,
        bucket=bucketfs_params.bucket)

