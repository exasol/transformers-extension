import ssl
import textwrap
from pathlib import Path
from urllib.parse import urlparse

import pyexasol
import pytest
from click.testing import CliRunner
from pyexasol import ExaConnection
from pytest_itde import config
from pytest_itde.config import TestConfig

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings


@revert_language_settings
def _call_deploy_language_container_deployer_cli(
        language_alias: str,
        schema: str,
        pyexasol_connection: ExaConnection,
        version, container_path,
        language_settings,
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs):
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    dsn = f"{exasol_config.host}:{exasol_config.port}"

    parsed_url = urlparse(bucketfs_config.url)
    # call language container deployer
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
        "--container-file", container_path,
        "--version", version,
        "--dsn", dsn,
        "--db-user", exasol_config.username,
        "--db-pass", exasol_config.password,
        "--language-alias", language_alias
    ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0

    # create a sample UDF using the new language alias
    db_conn_test = pyexasol.connect(
        dsn=dsn,
        user=exasol_config.username,
        password=exasol_config.password,
        encryption=True,
        websocket_sslopt={
            "cert_reqs": ssl.CERT_NONE,
        }
    )
    db_conn_test.execute(f"OPEN SCHEMA {schema}")
    db_conn_test.execute(textwrap.dedent(f"""
    CREATE OR REPLACE {language_alias} SCALAR SCRIPT "TEST_UDF"()
    RETURNS BOOLEAN AS

    def run(ctx):
        return True

    /
    """))
    result = db_conn_test.execute('SELECT "TEST_UDF"()').fetchall()
    return result


def test_language_container_deployer_cli_with_container_file(
        request,
        language_container,
        pyexasol_connection: ExaConnection,
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    result = _call_deploy_language_container_deployer_cli(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        pyexasol_connection=pyexasol_connection,
        container_path=Path(language_container["container_path"]),
        version=None,
        language_settings=language_settings,
        exasol_config=exasol_config,
        bucketfs_config=bucketfs_config
    )

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
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    result = _call_deploy_language_container_deployer_cli(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        pyexasol_connection=pyexasol_connection,
        container_path=None,
        version="0.2.0",
        language_settings=language_settings,
        exasol_config=exasol_config,
        bucketfs_config=bucketfs_config
    )

    assert result[0][0]


def test_language_container_deployer_cli_with_missing_container_option(
        request,
        pyexasol_connection: ExaConnection,
        exasol_config: config.Exasol,
        bucketfs_config: config.BucketFs
):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    with pytest.raises(Exception) as exc_info:
        _call_deploy_language_container_deployer_cli(
            language_alias="PYTHON3_TE",
            schema=schema_name,
            itde=itde,
            pyexasol_connection=pyexasol_connection,
            container_path=None,
            version=None,
            language_settings=language_settings,
            exasol_config=exasol_config,
            bucketfs_config=bucketfs_config
        )
