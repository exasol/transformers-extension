import textwrap
from pathlib import Path

from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension.deployment.language_container_deployer \
    import LanguageContainerDeployer
from tests.utils.db_queries import DBQueries
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings


@revert_language_settings
def _call_deploy_language_container_deployer(
        language_alias: str, schema: str, pyexasol_connection: ExaConnection,
        bucketfs_config: config.BucketFs, container_path, language_settings):
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

    # call language container deployer
    bucket_fs_factory = BucketFSFactory()
    bucketfs_location = bucket_fs_factory.create_bucketfs_location(
        url=f"{bucketfs_config.url}/"
            f"{bucketfs_params.bucket}/{bucketfs_params.path_in_bucket};"
            f"{bucketfs_params.name}",
        user=f"{bucketfs_config.username}",
        pwd=f"{bucketfs_config.password}",
        base_path=None)
    language_container_deployer = LanguageContainerDeployer(
        pyexasol_connection, language_alias, bucketfs_location, container_path)
    language_container_deployer.deploy_container()

    # create a sample UDF using the new language alias
    pyexasol_connection.execute(textwrap.dedent(f"""
    CREATE OR REPLACE {language_alias} SCALAR SCRIPT "TEST_UDF"()
    RETURNS BOOLEAN AS

    def run(ctx):
        return True

    /
    """))
    result = pyexasol_connection.execute('SELECT "TEST_UDF"()').fetchall()
    return result


def test_language_container_deployer(
        request, pyexasol_connection: ExaConnection, bucketfs_config: config.BucketFs, language_container):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    result = _call_deploy_language_container_deployer(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        pyexasol_connection=pyexasol_connection,
        bucketfs_config=bucketfs_config,
        container_path=Path(language_container["container_path"]),
        language_settings=language_settings
    )

    assert result[0][0]
