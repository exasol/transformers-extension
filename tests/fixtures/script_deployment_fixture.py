from __future__ import annotations
from typing import Any

import pytest
from pytest_itde import config
import exasol.bucketfs as bfs


@pytest.fixture(scope="session")
def deploy_params_onprem(exasol_config: config.Exasol) -> dict[str, Any]:
    return {
        'dsn': f"{exasol_config.host}:{exasol_config.port}",
        'db_user': exasol_config.username,
        'db_pass': exasol_config.password
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
    if backend == bfs.path.StorageBackend.onprem:
        yield deploy_params_onprem
    elif backend == bfs.path.StorageBackend.saas:
        yield deploy_params_saas
    else:
        raise ValueError(f'No deploy_params fixture for the backend {backend}')
