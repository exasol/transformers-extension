import textwrap
from typing import Optional, Callable
from urllib.parse import urlparse

import pytest
from _pytest.fixtures import FixtureRequest
from click.testing import CliRunner
from exasol_script_languages_container_tool.lib.tasks.export.export_info import ExportInfo
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension import deploy
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings


def assert_udf_running(language_alias: str,
                       schema: str,
                       connection_factory: Callable[[config.Exasol], ExaConnection],
                       exasol_config: config.Exasol):
    # We need a new connection to get the new system value for the SCRIPT_LANGUAGES parameter
    with connection_factory(exasol_config) as pyexasol_connection:
        pyexasol_connection.execute(f"OPEN SCHEMA {schema}")
        pyexasol_connection.execute(textwrap.dedent(f"""
        CREATE OR REPLACE {language_alias} SCALAR SCRIPT "TEST_UDF"()
        RETURNS BOOLEAN AS
    
        def run(ctx):
            return True
    
        /
        """))
        result = pyexasol_connection.execute('SELECT "TEST_UDF"()').fetchall()
        assert result[0][0] == True


def call_language_definition_deployer_cli(dsn: str,
                                          container_path: Optional[str],
                                          language_alias: str,
                                          version: Optional[str],
                                          exasol_config: config.Exasol,
                                          bucketfs_config: config.BucketFs,
                                          use_ssl: bool = False):
    parsed_url = urlparse(bucketfs_config.url)
    args_list = [
        "language-container",
        "--bucketfs-name", bucketfs_params.name,
        "--bucketfs-host", parsed_url.hostname,
        "--bucketfs-port", parsed_url.port,
        "--bucketfs_use-https", False,
        "--bucketfs-user", bucketfs_config.username,
        "--bucketfs-password", bucketfs_config.password,
        "--bucket", bucketfs_params.bucket,
        "--path-in-bucket", bucketfs_params.path_in_bucket,
        "--dsn", dsn,
        "--db-user", exasol_config.username,
        "--db-pass", exasol_config.password,
        "--language-alias", language_alias,
        "--use_ssl_cert", use_ssl
    ]
    if version is not None:
        args_list += [
            "--version", version,
        ]
    if container_path is not None:
        args_list += [
            "--container-file", container_path,
        ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    return result


def create_schema(pyexasol_connection: ExaConnection, schema: str):
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA {schema};")


def test_language_container_deployer_cli_with_container_file(
        request: FixtureRequest,
        export_slc: ExportInfo,
        pyexasol_connection: ExaConnection,
        connection_factory: Callable[[config.Exasol], ExaConnection],
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    test_name: str = request.node.name
    schema = test_name
    language_alias = f"PYTHON3_TE_{test_name.upper()}"
    container_path = export_slc.cache_file
    version = None
    create_schema(pyexasol_connection, schema)
    dsn = f"{exasol_config.host}:{exasol_config.port}"
    with revert_language_settings(pyexasol_connection):
        result = call_language_definition_deployer_cli(dsn=dsn,
                                                       container_path=container_path,
                                                       language_alias=language_alias,
                                                       version=version,
                                                       exasol_config=exasol_config,
                                                       bucketfs_config=bucketfs_config)
        assert result.exit_code == 0 and result.exception == None and result.stdout == ""
        assert_udf_running(connection_factory=connection_factory,
                           exasol_config=exasol_config,
                           language_alias=language_alias,
                           schema=schema)


def test_language_container_deployer_cli_by_downloading_container(
        request: FixtureRequest,
        pyexasol_connection: ExaConnection,
        connection_factory: Callable[[config.Exasol], ExaConnection],
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    test_name: str = request.node.name
    schema = test_name
    language_alias = f"PYTHON3_TE_{test_name.upper()}"
    container_path = None
    version = "0.2.0"
    create_schema(pyexasol_connection, schema)
    dsn = f"{exasol_config.host}:{exasol_config.port}"
    with revert_language_settings(pyexasol_connection):
        result = call_language_definition_deployer_cli(
            dsn=dsn,
            container_path=container_path,
            language_alias=language_alias,
            version=version,
            bucketfs_config=bucketfs_config,
            exasol_config=exasol_config
        )
        assert result.exit_code == 0 and result.exception == None and result.stdout == ""
        assert_udf_running(connection_factory=connection_factory,
                           exasol_config=exasol_config,
                           language_alias=language_alias,
                           schema=schema)


def test_language_container_deployer_cli_with_missing_container_option(
        request: FixtureRequest,
        pyexasol_connection: ExaConnection,
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    test_name: str = request.node.name
    language_alias = f"PYTHON3_TE_{test_name.upper()}"
    dsn = f"{exasol_config.host}:{exasol_config.port}"
    with revert_language_settings(pyexasol_connection):
        result = call_language_definition_deployer_cli(
            dsn=dsn,
            container_path=None,
            language_alias=language_alias,
            version=None,
            bucketfs_config=bucketfs_config,
            exasol_config=exasol_config
        )
        expected_exception_message = "You should specify either the release version to " \
                                     "download container file or the path of the already " \
                                     "downloaded container file."
        assert result.exit_code == 1 \
               and result.exception.args[0] == expected_exception_message \
               and type(result.exception) == ValueError

def assert_encryption_used():

    pass


def test_language_container_deployer_cli_with_use_SSL(
        request: FixtureRequest,
        export_slc: ExportInfo,
        pyexasol_connection: ExaConnection,
        connection_factory: Callable[[config.Exasol], ExaConnection],
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    use_ssl = True
    test_name: str = request.node.name
    schema = test_name
    language_alias = f"PYTHON3_TE_{test_name.upper()}"
    #container_path = export_slc.cache_file
    version = None
    create_schema(pyexasol_connection, schema)
    dsn = f"{exasol_config.host}:{exasol_config.port}"
    with revert_language_settings(pyexasol_connection):
        result = call_language_definition_deployer_cli(dsn=dsn,
                                                       container_path=None,
                                                       language_alias=language_alias,
                                                       version=version,
                                                       exasol_config=exasol_config,
                                                       bucketfs_config=bucketfs_config,
                                                       use_ssl=use_ssl)
        assert result.exit_code == 0 and result.exception == None and result.stdout == ""
        assert_udf_running(connection_factory=connection_factory,
                           exasol_config=exasol_config,
                           language_alias=language_alias,
                           schema=schema)
        assert_encryption_used()
