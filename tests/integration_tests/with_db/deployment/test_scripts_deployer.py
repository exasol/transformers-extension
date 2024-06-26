from _pytest.fixtures import FixtureRequest
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.db_queries import DBQueries


def test_scripts_deployer(
        language_alias: str,
        pyexasol_connection: ExaConnection,
        exasol_config: config.Exasol,
        request: FixtureRequest):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    ScriptsDeployer.run(
        dsn=f"{exasol_config.host}:{exasol_config.port}",
        db_user=exasol_config.username,
        db_pass=exasol_config.password,
        schema=schema_name,
        language_alias=language_alias,
        ssl_trusted_ca="",
        use_ssl_cert_validation=False
    )
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)


def test_scripts_deployer_no_schema_creation_permission(
        language_alias: str,
        pyexasol_connection,
        exasol_config: config.Exasol,
        request: FixtureRequest):
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
        language_alias=language_alias,
        ssl_trusted_ca="",
        use_ssl_cert_validation=False
    )
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)
