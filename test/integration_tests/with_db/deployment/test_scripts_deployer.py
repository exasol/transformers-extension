from __future__ import annotations

from test.utils.db_queries import (
    DBQueries,
    expected_script_list_all,
    expected_script_list_with_span,
    expected_script_list_without_span,
)
from typing import Any

import exasol.bucketfs as bfs
import pytest
from exasol.pytest_backend import BACKEND_ONPREM
from exasol.python_extension_common.deployment.language_container_validator import (
    temp_schema,
)
from pyexasol import ExaConnection

from exasol_transformers_extension.deployment.scripts_deployer import ScriptsDeployer


def test_scripts_deployer(
    database_std_params,
    pyexasol_connection: ExaConnection,
    language_alias,
    deployed_slc,
):

    with temp_schema(pyexasol_connection) as schema_name:
        ScriptsDeployer.run(
            **database_std_params, schema=schema_name, language_alias=language_alias
        )
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name, expected_script_list_without_span
        )


def test_scripts_deployer_all_scripts(
    database_std_params,
    pyexasol_connection: ExaConnection,
    language_alias,
    deployed_slc,
):

    with temp_schema(pyexasol_connection) as schema_name:
        ScriptsDeployer.run(
            **database_std_params,
            schema=schema_name,
            language_alias=language_alias,
            install_all_scripts=True,
        )
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name, expected_script_list_all
        )


def test_scripts_deployer_with_span(
    database_std_params,
    pyexasol_connection: ExaConnection,
    language_alias,
    deployed_slc,
):

    with temp_schema(pyexasol_connection) as schema_name:
        ScriptsDeployer.run(
            **database_std_params,
            schema=schema_name,
            language_alias=language_alias,
            use_spans=True,
        )
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name, expected_script_list_with_span
        )


def test_scripts_deployer_no_schema_creation_permission(
    backend, database_std_params, pyexasol_connection, language_alias, deployed_slc
):

    if backend != BACKEND_ONPREM:
        pytest.skip(
            "We run this test only with the Docker-DB, "
            "since the script deployer doesn't use the DB user login and password in SaaS."
        )

    with temp_schema(pyexasol_connection) as schema_name:
        limited_user = "limited_user"
        limited_user_password = "limited_user"
        pyexasol_connection.execute(f"DROP USER IF EXISTS {limited_user};")
        pyexasol_connection.execute(
            f"""CREATE USER {limited_user} IDENTIFIED BY "{limited_user_password}";"""
        )
        for permission in [
            "CREATE SESSION",
            "CREATE TABLE",
            "CREATE ANY TABLE",
            "SELECT ANY TABLE",
            "SELECT ANY DICTIONARY",
            "CREATE VIEW",
            "CREATE ANY VIEW",
            "CREATE SCRIPT",
            "CREATE ANY SCRIPT",
            "EXECUTE ANY SCRIPT",
            "USE ANY SCHEMA",
            "CREATE CONNECTION",
        ]:
            pyexasol_connection.execute(f"GRANT {permission} TO {limited_user}; ")

        ScriptsDeployer.run(
            dsn=database_std_params["dsn"],
            db_user=limited_user,
            db_pass=limited_user_password,
            schema=schema_name,
            language_alias=language_alias,
            ssl_trusted_ca="",
            use_ssl_cert_validation=False,
        )
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name, expected_script_list_without_span
        )
