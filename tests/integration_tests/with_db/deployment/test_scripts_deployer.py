from __future__ import annotations
from typing import Any

import pytest
from pyexasol import ExaConnection
import exasol.bucketfs as bfs
from exasol.python_extension_common.deployment.language_container_validator import temp_schema
from exasol.pytest_backend import BACKEND_ONPREM

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.db_queries import DBQueries
from tests.fixtures.language_container_fixture_constants import LANGUAGE_ALIAS


def test_scripts_deployer(
        backend,
        deploy_params: dict[str, Any],
        pyexasol_connection: ExaConnection,
        upload_slc):

    with temp_schema(pyexasol_connection) as schema_name:
        # We validate the server certificate in SaaS, but not in the Docker DB
        cert_validation = backend == bfs.path.StorageBackend.saas
        ScriptsDeployer.run(**deploy_params,
                            schema=schema_name,
                            language_alias=LANGUAGE_ALIAS,
                            use_ssl_cert_validation=cert_validation)
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name)


def test_scripts_deployer_no_schema_creation_permission(
        backend,
        deploy_params,
        pyexasol_connection,
        upload_slc):

    if backend != BACKEND_ONPREM:
        pytest.skip(("We run this test only with the Docker-DB, "
                     "since the script deployer doesn't use the DB user login and password in SaaS."))

    with temp_schema(pyexasol_connection) as schema_name:
        limited_user = "limited_user"
        limited_user_password = "limited_user"
        pyexasol_connection.execute(f"DROP USER IF EXISTS {limited_user};")
        pyexasol_connection.execute(f"""CREATE USER {limited_user} IDENTIFIED BY "{limited_user_password}";""")
        for permission in ["CREATE SESSION", "CREATE TABLE", "CREATE ANY TABLE", "SELECT ANY TABLE",
                           "SELECT ANY DICTIONARY", "CREATE VIEW", "CREATE ANY VIEW", "CREATE SCRIPT", "CREATE ANY SCRIPT",
                           "EXECUTE ANY SCRIPT", "USE ANY SCHEMA", "CREATE CONNECTION"]:
            pyexasol_connection.execute(f"GRANT {permission} TO {limited_user}; ")

        ScriptsDeployer.run(
            dsn=deploy_params['dsn'],
            db_user=limited_user,
            db_pass=limited_user_password,
            schema=schema_name,
            language_alias=LANGUAGE_ALIAS,
            ssl_trusted_ca="",
            use_ssl_cert_validation=False
        )
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name)
