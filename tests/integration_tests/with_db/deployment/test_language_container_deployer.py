import textwrap
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from pytest_itde.config import TestConfig

from exasol_transformers_extension.deployment.language_container_deployer \
    import LanguageContainerDeployer
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings
from tests.utils.db_queries import DBQueries
from pathlib import Path


@revert_language_settings
def _call_deploy_language_container_deployer(
        language_alias, schema, itde: TestConfig, container_path, language_settings):
    itde.ctrl_connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    itde.ctrl_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

    # call language container deployer
    bucket_fs_factory = BucketFSFactory()
    bucketfs_location = bucket_fs_factory.create_bucketfs_location(
        url=f"{itde.bucketfs.url}/"
            f"{bucketfs_params.bucket}/{bucketfs_params.path_in_bucket};"
            f"{bucketfs_params.name}",
        user=f"{itde.bucketfs.username}",
        pwd=f"{itde.bucketfs.password}",
        base_path=None)
    language_container_deployer = LanguageContainerDeployer(
        itde.ctrl_connection, language_alias, bucketfs_location, container_path)
    language_container_deployer.deploy_container()

    # create a sample UDF using the new language alias
    itde.ctrl_connection.execute(textwrap.dedent(f"""
    CREATE OR REPLACE {language_alias} SCALAR SCRIPT "TEST_UDF"()
    RETURNS BOOLEAN AS

    def run(ctx):
        return True

    /
    """))
    result = itde.ctrl_connection.execute('SELECT "TEST_UDF"()').fetchall()
    return result


def test_language_container_deployer(
        request, itde, language_container):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(itde.ctrl_connection)

    result = _call_deploy_language_container_deployer(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        itde=itde,
        container_path=Path(language_container["container_path"]),
        language_settings=language_settings
    )

    assert result[0][0]
