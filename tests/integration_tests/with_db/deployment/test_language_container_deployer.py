import textwrap
from pathlib import Path

from _pytest.fixtures import FixtureRequest
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_script_languages_container_tool.lib.tasks.export.export_info import ExportInfo
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension.deployment.language_container_deployer \
    import LanguageContainerDeployer
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings


def test_language_container_deployer(
        request: FixtureRequest,
        export_slc: ExportInfo,
        pyexasol_connection: ExaConnection,
        bucketfs_config: config.BucketFs,
):
    test_name: str = request.node.name
    schema = test_name
    language_alias = f"PYTHON3_TE_{test_name.upper()}"
    container_path = Path(export_slc.cache_file)
    with revert_language_settings(pyexasol_connection):
        create_schema(pyexasol_connection, schema)
        call_language_container_deployer(container_path=container_path,
                                         language_alias=language_alias,
                                         pyexasol_connection=pyexasol_connection,
                                         bucketfs_config=bucketfs_config)
        assert_udf_running(pyexasol_connection, language_alias)


def create_schema(pyexasol_connection: ExaConnection, schema: str):
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")


def assert_udf_running(pyexasol_connection: ExaConnection, language_alias: str):
    pyexasol_connection.execute(textwrap.dedent(f"""
        CREATE OR REPLACE {language_alias} SCALAR SCRIPT "TEST_UDF"()
        RETURNS BOOLEAN AS
        def run(ctx):
            return True
        /
        """))
    result = pyexasol_connection.execute('SELECT "TEST_UDF"()').fetchall()
    assert result[0][0] == True


def call_language_container_deployer(container_path: Path,
                                     language_alias: str,
                                     pyexasol_connection: ExaConnection,
                                     bucketfs_config: config.BucketFs):
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
