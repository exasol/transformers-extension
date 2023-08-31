from click.testing import CliRunner
from pyexasol import ExaConnection, ExaRequestError
from pytest_itde import config

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
        "--language-alias", language_alias
    ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)


def test_scripts_deployer_cli_with_encryption_verfiy(language_alias: str,
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
        "--use_ssl_cert_validation", True
    ]
    expected_exception_message = 'Connection exception - Client connection must be encrypted.'
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 1 \
           and result.exception.args[0] == expected_exception_message \
           and type(result.exception) == ExaRequestError
