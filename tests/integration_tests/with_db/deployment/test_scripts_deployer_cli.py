from click.testing import CliRunner
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries


def test_scripts_deployer_cli(upload_language_container: str,
                              pyexasol_connection: ExaConnection,
                              exasol_config: config.Exasol,
                              request):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    language_alias = "PYTHON3_TE"
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
