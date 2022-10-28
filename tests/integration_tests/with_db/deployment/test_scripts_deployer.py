from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.db_queries import DBQueries
from tests.utils.parameters import db_params


def test_scripts_deployer(upload_language_container,
                          pyexasol_connection, request):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    language_alias = upload_language_container
    ScriptsDeployer.run(
        dsn=db_params.address(),
        user=db_params.user,
        password=db_params.password,
        schema=schema_name,
        language_alias=language_alias
    )
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)


def test_scripts_deployer_no_schema_creation_permission(upload_language_container,
                                                        pyexasol_connection, request):
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

    language_alias = upload_language_container
    ScriptsDeployer.run(
        dsn=db_params.address(),
        user=limited_user,
        password=limited_user_password,
        schema=schema_name,
        language_alias=language_alias
    )
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)
