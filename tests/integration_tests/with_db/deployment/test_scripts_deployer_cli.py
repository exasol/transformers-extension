from click.testing import CliRunner

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries


def test_scripts_deployer_cli(upload_language_container,
                              itde,
                              request):
    schema_name = request.node.name
    itde.ctrl_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    language_alias = "PYTHON3_TE"
    args_list = [
        "scripts",
        "--dsn", f"{itde.db.host}:{itde.db.port}",
        "--db-user", itde.db.username,
        "--db-pass", itde.db.password,
        "--schema", schema_name,
        "--language-alias", language_alias
    ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0
    assert DBQueries.check_all_scripts_deployed(
        itde.ctrl_connection, schema_name)
