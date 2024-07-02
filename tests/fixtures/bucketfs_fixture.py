from __future__ import annotations

import pytest
from pytest_itde.config import TestConfig
import exasol.bucketfs as bfs

from exasol_transformers_extension.utils.bucketfs_operations import create_bucketfs_location
from tests.utils.parameters import bucketfs_params


@pytest.fixture(scope="session")
def bucketfs_location_onprem(backend,
                             itde: TestConfig) -> bfs.path.PathLike | None:

    if backend == bfs.path.StorageBackend.onprem:
        return create_bucketfs_location(
            path_in_bucket=bucketfs_params.path_in_bucket,
            bucketfs_name=bucketfs_params.name,
            bucketfs_url=itde.bucketfs.url,
            bucketfs_user=itde.bucketfs.username,
            bucketfs_password=itde.bucketfs.password,
            bucket=bucketfs_params.bucket)
    return None


@pytest.fixture(scope="session")
def bucketfs_location_saas(backend,
                           saas_url,
                           saas_account_id,
                           saas_database_id,
                           saas_token) -> bfs.path.PathLike | None:

    if backend == bfs.path.StorageBackend.saas:
        return create_bucketfs_location(
            path_in_bucket=bucketfs_params.path_in_bucket,
            saas_url=saas_url,
            saas_account_id=saas_account_id,
            saas_database_id=saas_database_id,
            saas_token=saas_token)
    return None


@pytest.fixture(scope="session")
def bucketfs_location(backend,
                      bucketfs_location_onprem,
                      bucketfs_location_saas) -> bfs.path.PathLike:

    if backend == bfs.path.StorageBackend.onprem:
        assert bucketfs_location_onprem is not None
        return bucketfs_location_onprem
    elif backend == bfs.path.StorageBackend.saas:
        assert bucketfs_location_saas is not None
        return bucketfs_location_saas
    else:
        raise ValueError(f'No bucketfs_location fixture for the backend {backend}')
