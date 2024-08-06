from __future__ import annotations
import os
from datetime import timedelta
from contextlib import ExitStack

import ssl
import pyexasol
import pytest
import exasol.bucketfs as bfs
from _pytest.fixtures import FixtureRequest
from exasol.saas.client.api_access import (
    OpenApiAccess,
    create_saas_client,
    timestamp_name,
    get_connection_params
)
from pytest_itde import config

CURRENT_SAAS_DATABASE_ID = pytest.StashKey[str]()


def _env(var: str) -> str:
    result = os.environ.get(var)
    if result:
        return result
    raise RuntimeError(f"Environment variable {var} is empty.")


_BACKEND_OPTION = '--backend'
BACKEND_ONPREM = 'onprem'
BACKEND_SAAS = 'saas'


@pytest.fixture(scope='session', params=[BACKEND_ONPREM, BACKEND_SAAS])
def backend(request) -> str:
    backend_options = request.config.getoption(_BACKEND_OPTION)
    if backend_options and (request.param not in backend_options):
        pytest.skip()
    return request.param


@pytest.fixture(scope="session")
def saas_url(backend) -> str:
    if backend == BACKEND_SAAS:
        return _env("SAAS_HOST")


@pytest.fixture(scope="session")
def saas_account_id(backend) -> str:
    if backend == BACKEND_SAAS:
        return _env("SAAS_ACCOUNT_ID")


@pytest.fixture(scope="session")
def saas_token(backend) -> str:
    if backend == BACKEND_SAAS:
        return _env("SAAS_PAT")



@pytest.fixture(scope="session")
def saas_database_id(request: FixtureRequest, backend, saas_url, saas_account_id, saas_token) -> str:
    if backend == BACKEND_SAAS:
        if CURRENT_SAAS_DATABASE_ID in request.session.stash:
            with ExitStack() as stack:
                # Create and configure the SaaS client.
                client = create_saas_client(host=saas_url, pat=saas_token)
                api_access = OpenApiAccess(client=client, account_id=saas_account_id)
                stack.enter_context(api_access.allowed_ip())

                # Create a temporary database and waite till it becomes operational
                db = stack.enter_context(api_access.database(
                    name=timestamp_name('TE_CI'),
                    idle_time=timedelta(hours=12)))
                api_access.wait_until_running(db.id)
                request.session.stash[CURRENT_SAAS_DATABASE_ID] = db.id
                yield db.id
        else:
            yield request.session.stash[CURRENT_SAAS_DATABASE_ID]
    else:
        yield ''


@pytest.fixture(scope="session")
def pyexasol_connection_onprem(backend,
                               connection_factory,
                               exasol_config: config.Exasol) -> pyexasol.ExaConnection | None:
    if backend == BACKEND_ONPREM:
        with connection_factory(exasol_config) as conn:
            yield conn
    else:
        yield None


@pytest.fixture(scope="session")
def pyexasol_connection_saas(backend,
                             saas_url,
                             saas_account_id,
                             saas_database_id,
                             saas_token) -> pyexasol.ExaConnection | None:
    if backend == BACKEND_SAAS:
        # Create a connection to the database.
        conn_params = get_connection_params(host=saas_url,
                                            account_id=saas_account_id,
                                            database_id=saas_database_id,
                                            pat=saas_token)
        with pyexasol.connect(**conn_params,
                              encryption=True,
                              websocket_sslopt={"cert_reqs": ssl.CERT_NONE},
                              compression=True) as conn:
            yield conn
    else:
        yield None


@pytest.fixture(scope="session")
def pyexasol_connection(backend,
                        pyexasol_connection_onprem,
                        pyexasol_connection_saas) -> pyexasol.ExaConnection:
    if backend == BACKEND_ONPREM:
        assert pyexasol_connection_onprem is not None
        yield pyexasol_connection_onprem
    elif backend == BACKEND_SAAS:
        assert pyexasol_connection_saas is not None
        yield pyexasol_connection_saas
    else:
        raise ValueError(f'No pyexasol_connection fixture for the backend {backend}')
