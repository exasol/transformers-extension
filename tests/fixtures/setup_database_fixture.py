from typing import Tuple
import json

from urllib.parse import urlparse

import pyexasol
import pytest
from pyexasol import ExaConnection
from pytest_itde import config
import exasol.bucketfs as bfs

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.fixtures.database_connection_fixture_constants import BACKEND_ONPREM, BACKEND_SAAS
from tests.utils.parameters import bucketfs_params
from tests.fixtures.language_container_fixture_constants import LANGUAGE_ALIAS

BUCKETFS_CONNECTION_NAME = "TEST_TE_BFS_CONNECTION"
SCHEMA_NAME = "TEST_INTEGRATION"


def _create_schema(pyexasol_connection: ExaConnection) -> None:
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA {SCHEMA_NAME};")
    pyexasol_connection.execute(f"OPEN SCHEMA {SCHEMA_NAME};")


def _deploy_scripts(pyexasol_connection: ExaConnection) -> None:
    scripts_deployer = ScriptsDeployer(schema=SCHEMA_NAME,
                                       language_alias=LANGUAGE_ALIAS,
                                       pyexasol_conn=pyexasol_connection)
    scripts_deployer.deploy_scripts()
    print("_deploy_scripts CURRENT_SESSION:", pyexasol_connection.execute("SELECT CURRENT_SESSION").fetchall())
    print("_deploy_scripts CURRENT_SCHEMA:", pyexasol_connection.execute("SELECT CURRENT_SCHEMA").fetchall())
    print("_deploy_scripts schema:", SCHEMA_NAME)


def _to_json_str(**kwargs) -> str:
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return json.dumps(filtered_kwargs)


def _create_bucketfs_connection(pyexasol_connection: ExaConnection,
                                conn_to: str,
                                conn_user: str,
                                conn_password: str) -> None:

    query = (f"CREATE OR REPLACE  CONNECTION {BUCKETFS_CONNECTION_NAME} "
             f"TO '{conn_to}' "
             f"USER '{conn_user}' "
             f"IDENTIFIED BY '{conn_password}'")
    pyexasol_connection.execute(query)


def _create_bucketfs_connection_onprem(bucketfs_config: config.BucketFs,
                                       pyexasol_connection: ExaConnection) -> None:

    parsed_url = urlparse(bucketfs_config.url)
    host = parsed_url.netloc.split(":")[0]
    url = f"{parsed_url.scheme}://{host}:{bucketfs_params.real_port}"
    conn_to = _to_json_str(backend=bfs.path.StorageBackend.onprem.name,
                           url=url,
                           service_name=bucketfs_params.name,
                           bucket_name=bucketfs_params.bucket,
                           path=bucketfs_params.path_in_bucket,
                           verify=False)
    conn_user = _to_json_str(username=bucketfs_config.username)
    conn_password = _to_json_str(password=bucketfs_config.password)

    _create_bucketfs_connection(pyexasol_connection, conn_to, conn_user, conn_password)


def _create_bucketfs_connection_saas(url: str,
                                     account_id: str,
                                     database_id: str,
                                     token: str,
                                     pyexasol_connection: ExaConnection) -> None:

    conn_to = _to_json_str(backend=bfs.path.StorageBackend.saas.name,
                           url=url,
                           path=bucketfs_params.path_in_bucket)
    conn_user = _to_json_str(account_id=account_id,
                             database_id=database_id)
    conn_password = _to_json_str(pat=token)

    _create_bucketfs_connection(pyexasol_connection, conn_to, conn_user, conn_password)


@pytest.fixture(scope="session")
def setup_database(backend: bfs.path.StorageBackend,
                   bucketfs_config: config.BucketFs,
                   saas_url: str,
                   saas_account_id: str,
                   saas_database_id: str,
                   saas_token: str,
                   pyexasol_connection: ExaConnection,
                   upload_slc) -> Tuple[str, str]:

    _create_schema(pyexasol_connection)
    _deploy_scripts(pyexasol_connection)
    if backend == BACKEND_ONPREM:
        _create_bucketfs_connection_onprem(bucketfs_config, pyexasol_connection)
    elif backend == BACKEND_SAAS:
        _create_bucketfs_connection_saas(saas_url, saas_account_id, saas_database_id, saas_token,
                                         pyexasol_connection)
    else:
        raise ValueError(f'No setup_database fixture for the backend {backend}')

    return BUCKETFS_CONNECTION_NAME, SCHEMA_NAME


@pytest.fixture()
def db_conn(setup_database, pyexasol_connection) -> pyexasol.ExaConnection:
    """
    Per-test fixture that returns the same session-wide pyexasol connection,
    but makes sure the default schema is open.
    """
    pyexasol_connection.execute(f"OPEN SCHEMA {SCHEMA_NAME};")
    return pyexasol_connection
