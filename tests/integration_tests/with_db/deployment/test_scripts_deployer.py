from pytest_itde.config import TestConfig

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.db_queries import DBQueries


def test_scripts_deployer(
        upload_language_container,
        itde,
        request):
    schema_name = request.node.name
    itde.ctrl_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    language_alias = upload_language_container
    ScriptsDeployer.run(
        dsn=f"{itde.db.host}:{itde.db.port}",
        user=itde.db.username,
        password=itde.db.password,
        schema=schema_name,
        language_alias=language_alias
    )
    assert DBQueries.check_all_scripts_deployed(
        itde.ctrl_connection, schema_name)


def test_scripts_deployer_no_schema_creation_permission(
        upload_language_container,
        itde: TestConfig,
        request):
    schema_name = request.node.name
    itde.ctrl_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
    itde.ctrl_connection.execute(f"CREATE SCHEMA {schema_name};")

    limited_user = "limited_user"
    limited_user_password = "limited_user"
    itde.ctrl_connection.execute(f"DROP USER IF EXISTS {limited_user};")
    itde.ctrl_connection.execute(f"""CREATE USER {limited_user} IDENTIFIED BY "{limited_user_password}";""")
    for permission in ["CREATE SESSION", "CREATE TABLE", "CREATE ANY TABLE", "SELECT ANY TABLE",
                       "SELECT ANY DICTIONARY", "CREATE VIEW", "CREATE ANY VIEW", "CREATE SCRIPT", "CREATE ANY SCRIPT",
                       "EXECUTE ANY SCRIPT", "USE ANY SCHEMA", "CREATE CONNECTION"]:
        itde.ctrl_connection.execute(f"GRANT {permission} TO {limited_user}; ")

    language_alias = upload_language_container
    ScriptsDeployer.run(
        dsn=f"{itde.db.host}:{itde.db.port}",
        user=limited_user,
        password=limited_user_password,
        schema=schema_name,
        language_alias=language_alias
    )
    assert DBQueries.check_all_scripts_deployed(
        itde.ctrl_connection, schema_name)
