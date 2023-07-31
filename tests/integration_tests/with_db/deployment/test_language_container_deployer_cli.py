import ssl
import textwrap
from typing import Optional
from urllib.parse import urlparse

import pyexasol
import pytest
from click.testing import CliRunner
from exasol_script_languages_container_tool.lib.tasks.export.export_info import ExportInfo
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension import deploy
from tests.utils.parameters import bucketfs_params


def create_and_run_test_udf(language_alias: str,
                            schema: str,
                            pyexasol_connection: ExaConnection):
    pyexasol_connection.execute(f"OPEN SCHEMA {schema}")
    pyexasol_connection.execute(textwrap.dedent(f"""
    CREATE OR REPLACE {language_alias} SCALAR SCRIPT "TEST_UDF"()
    RETURNS BOOLEAN AS

    def run(ctx):
        return True

    /
    """))
    result = pyexasol_connection.execute('SELECT "TEST_UDF"()').fetchall()
    return result


def call_language_definition_deployer_cli(dsn: str,
                                          container_path: Optional[str],
                                          language_alias: str,
                                          version: Optional[str],
                                          exasol_config: config.Exasol,
                                          bucketfs_config: config.BucketFs):
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
        "--language-alias", language_alias
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
    pyexasol_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")


def test_language_container_deployer_cli_with_container_file(
        request,
        export_slc: ExportInfo,
        pyexasol_connection: ExaConnection,
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
    result = call_language_definition_deployer_cli(dsn=dsn,
                                                   container_path=container_path,
                                                   language_alias=language_alias,
                                                   version=version,
                                                   exasol_config=exasol_config,
                                                   bucketfs_config=bucketfs_config)
    assert result.exit_code == 0
    result = create_and_run_test_udf(pyexasol_connection=pyexasol_connection,
                                     language_alias=language_alias,
                                     schema=schema)
    assert result[0][0]


@pytest.mark.skip(reason="It causes this error:  error:  BucketFS: root path "
                         "'container/language_container'' does not exist in "
                         "bucket 'default' of bucketfs 'bfsdefault'.")
def test_language_container_deployer_cli_by_downloading_container(
        request,
        pyexasol_connection: ExaConnection,
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
    result = call_language_definition_deployer_cli(
        dsn=dsn,
        container_path=container_path,
        language_alias=language_alias,
        version=version,
        bucketfs_config=bucketfs_config,
        exasol_config=exasol_config
    )
    assert result.exit_code == 0
    result = create_and_run_test_udf(pyexasol_connection=pyexasol_connection,
                                     language_alias=language_alias,
                                     schema=schema)
    assert result[0][0]


def test_language_container_deployer_cli_with_missing_container_option(
        request,
        pyexasol_connection: ExaConnection,
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    test_name: str = request.node.name
    language_alias = f"PYTHON3_TE_{test_name.upper()}"
    dsn = f"{exasol_config.host}:{exasol_config.port}"
    result = call_language_definition_deployer_cli(
        dsn=dsn,
        container_path=container_path,
        language_alias=language_alias,
        version=version,
        bucketfs_config=bucketfs_config,
        exasol_config=exasol_config
    )
    assert result.exit_code == 0
    expected_exception_message = "You should specify either the release version to " \
                                 "download container file or the path of the already " \
                                 "downloaded container file."
    assert result.exit_code != 0 \
           and result.exception.args[0] == expected_exception_message \
           and type(result.exception) == ValueError
