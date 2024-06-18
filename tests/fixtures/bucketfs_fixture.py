from urllib.parse import urlparse

import pytest
from pytest_itde.config import TestConfig
import exasol.bucketfs as bfs

from exasol_transformers_extension.utils.bucketfs_operations import create_bucketfs_location
from tests.utils.parameters import bucketfs_params


@pytest.fixture(scope="session")
def bucketfs_location(itde: TestConfig) -> bfs.path.PathLike:
    return create_bucketfs_location(
        path_in_bucket=bucketfs_params.path_in_bucket,
        bucketfs_name=bucketfs_params.name,
        bucketfs_url=itde.bucketfs.url,
        bucketfs_user=itde.bucketfs.username,
        bucketfs_password=itde.bucketfs.password,
        bucket=bucketfs_params.bucket)
