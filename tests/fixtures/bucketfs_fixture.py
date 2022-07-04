import pytest
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from tests.utils.parameters import bucketfs_params
from exasol_bucketfs_utils_python.bucket_config import BucketConfig
from exasol_bucketfs_utils_python.bucketfs_config import BucketFSConfig
from exasol_bucketfs_utils_python.bucketfs_connection_config import \
    BucketFSConnectionConfig


@pytest.fixture(scope="session")
def bucket_config() -> BucketConfig:
    connection_config = BucketFSConnectionConfig(
        host=bucketfs_params.host, port=int(bucketfs_params.port),
        user=f"{bucketfs_params.user}", pwd=f"{bucketfs_params.password}",
        is_https=False)
    bucketfs_config = BucketFSConfig(
        connection_config=connection_config, bucketfs_name="bfsdefault")
    bucket_config = BucketConfig(
        bucket_name="default", bucketfs_config=bucketfs_config)
    return bucket_config


@pytest.fixture(scope="session")
def bucketfs_location() -> BucketFSLocation:
    return BucketFSFactory().create_bucketfs_location(
        url=bucketfs_params.address(),
        user=bucketfs_params.user,
        pwd=bucketfs_params.password)
