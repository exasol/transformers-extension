from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import pytest
from exasol.pytest_itde import config

from tests.fixtures.database_connection_fixture_constants import BACKEND_ONPREM, BACKEND_SAAS
from tests.utils.parameters import bucketfs_params


@pytest.fixture(scope="session")
def deploy_params_onprem(exasol_config: config.Exasol) -> dict[str, Any]:
    return {
        'dsn': f"{exasol_config.host}:{exasol_config.port}",
        'db_user': exasol_config.username,
        'db_pass': exasol_config.password
    }


@pytest.fixture(scope="session")
def upload_params_onprem(bucketfs_config: config.BucketFs):
    parsed_url = urlparse(bucketfs_config.url)
    host, port = parsed_url.netloc.split(":")
    return {
        "bucketfs-name": bucketfs_params.name,
        "bucketfs-host": host,
        "bucketfs-port": port,
        "bucketfs-use-https": False,
        "bucketfs-user": bucketfs_config.username,
        "bucketfs-password": bucketfs_config.password,
        "bucket": bucketfs_params.bucket
    }


@pytest.fixture(scope="session")
def deploy_params_saas(saas_url, saas_account_id, saas_database_id, saas_token) -> dict[str, Any]:
    yield {
        'saas_url': saas_url,
        'saas_account_id': saas_account_id,
        'saas_database_id': saas_database_id,
        'saas_token': saas_token
    }


@pytest.fixture(scope="session")
def deploy_params(backend,
                  deploy_params_onprem,
                  deploy_params_saas) -> dict[str, Any]:
    if backend == BACKEND_ONPREM:
        yield deploy_params_onprem
    elif backend == BACKEND_SAAS:
        yield deploy_params_saas
    else:
        raise ValueError(f'No deploy_params fixture for the backend {backend}')


@pytest.fixture(scope="session")
def upload_params(backend,
                  upload_params_onprem,
                  deploy_params_saas) -> dict[str, Any]:
    if backend == BACKEND_ONPREM:
        yield upload_params_onprem
    elif backend == BACKEND_SAAS:
        yield deploy_params_saas
    else:
        raise ValueError(f'No deploy_params fixture for the backend {backend}')
