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
