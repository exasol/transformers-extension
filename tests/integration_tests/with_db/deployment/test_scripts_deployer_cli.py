from click.testing import CliRunner
from pyexasol import ExaConnection, ExaConnectionFailedError
from pytest_itde import config

import pytest
from tests.fixtures.language_container_fixture import export_slc, flavor_path, language_alias
from tests.fixtures.database_connection_fixture import pyexasol_connection

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries


def test_scripts_deployer_cli(language_alias: str,
                              pyexasol_connection: ExaConnection,
                              exasol_config: config.Exasol,
                              request):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    args_list = [
        "scripts",
        "--dsn", f"{exasol_config.host}:{exasol_config.port}",
        "--db-user", exasol_config.username,
        "--db-pass", exasol_config.password,
        "--schema", schema_name,
        "--language-alias", language_alias,
        "--no-use-ssl-cert-validation"
    ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)


def test_scripts_deployer_cli_with_encryption_verify(language_alias: str,
                              pyexasol_connection: ExaConnection,
                              exasol_config: config.Exasol,
                              request):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    args_list = [
        "scripts",
        "--dsn", f"{exasol_config.host}:{exasol_config.port}",
        "--db-user", exasol_config.username,
        "--db-pass", exasol_config.password,
        "--schema", schema_name,
        "--language-alias", language_alias,
        "--use-ssl-cert-validation"
    ]
    expected_exception_message = '[SSL: CERTIFICATE_VERIFY_FAILED]'
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 1 \
           and expected_exception_message in result.exception.args[0].message \
           and type(result.exception) == ExaConnectionFailedError

