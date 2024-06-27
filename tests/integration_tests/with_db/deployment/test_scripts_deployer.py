from __future__ import annotations
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from pyexasol import ExaConnection
from pytest_itde import config
import exasol.bucketfs as bfs

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.db_queries import DBQueries
from tests.fixtures.language_container_fixture import LANGUAGE_ALIAS


def test_scripts_deployer(
        deploy_params: dict[str, Any],
        pyexasol_connection: ExaConnection,
        request: FixtureRequest):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    # We validate the server certificate in SaaS, but not in the Docker DB
    cert_validation = "saas_url" in deploy_params
    ScriptsDeployer.run(**deploy_params,
                        schema=schema_name,
                        language_alias=LANGUAGE_ALIAS,
                        use_ssl_cert_validation=cert_validation)
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)


def test_scripts_deployer_no_schema_creation_permission(
        backend,
        pyexasol_connection,
        exasol_config: config.Exasol,
        request: FixtureRequest):
    if backend != bfs.path.StorageBackend.onprem:
        pytest.skip("We run this test only in the Docker-DB")

    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA {schema_name};")

    limited_user = "limited_user"
    limited_user_password = "limited_user"
    pyexasol_connection.execute(f"DROP USER IF EXISTS {limited_user};")
    pyexasol_connection.execute(f"""CREATE USER {limited_user} IDENTIFIED BY "{limited_user_password}";""")
    for permission in ["CREATE SESSION", "CREATE TABLE", "CREATE ANY TABLE", "SELECT ANY TABLE",
                       "SELECT ANY DICTIONARY", "CREATE VIEW", "CREATE ANY VIEW", "CREATE SCRIPT", "CREATE ANY SCRIPT",
                       "EXECUTE ANY SCRIPT", "USE ANY SCHEMA", "CREATE CONNECTION"]:
        pyexasol_connection.execute(f"GRANT {permission} TO {limited_user}; ")

    ScriptsDeployer.run(
        dsn=f"{exasol_config.host}:{exasol_config.port}",
        db_user=limited_user,
        db_pass=limited_user_password,
        schema=schema_name,
        language_alias=LANGUAGE_ALIAS,
        ssl_trusted_ca="",
        use_ssl_cert_validation=False
    )
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)
