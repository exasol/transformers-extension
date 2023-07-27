import ssl
import textwrap
from pathlib import Path
from urllib.parse import urlparse

import pyexasol
import pytest
from click.testing import CliRunner
from pytest_itde.config import TestConfig

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings


@revert_language_settings
def _call_deploy_language_container_deployer_cli(
        language_alias,
        schema,
        itde: TestConfig,
        container_path,
        version,
        language_settings):
    itde.ctrl_connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    itde.ctrl_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    dsn = f"{itde.db.host}:{itde.db.port}"

    parsed_url = urlparse(itde.bucketfs.url)
    # call language container deployer
    args_list = [
        "language-container",
        "--bucketfs-name", bucketfs_params.name,
        "--bucketfs-host", parsed_url.hostname,
        "--bucketfs-port", parsed_url.port,
        "--bucketfs_use-https", False,
        "--bucketfs-user", itde.bucketfs.username,
        "--bucketfs-password", itde.bucketfs.password,
        "--bucket", bucketfs_params.bucket,
        "--path-in-bucket", bucketfs_params.path_in_bucket,
        "--container-file", container_path,
        "--version", version,
        "--dsn", dsn,
        "--db-user", itde.db.username,
        "--db-pass", itde.db.password,
        "--language-alias", language_alias
    ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0

    # create a sample UDF using the new language alias
    db_conn_test = pyexasol.connect(
        dsn=dsn,
        user=itde.db.username,
        password=itde.db.password,
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
        itde,
        language_container):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(itde.ctrl_connection)

    result = _call_deploy_language_container_deployer_cli(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        itde=itde,
        container_path=Path(language_container["container_path"]),
        version=None,
        language_settings=language_settings
    )

    assert result[0][0]


@pytest.mark.skip(reason="It causes this error:  error:  BucketFS: root path "
                         "'container/language_container'' does not exist in "
                         "bucket 'default' of bucketfs 'bfsdefault'.")
def test_language_container_deployer_cli_by_downloading_container(
        request, itde):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(itde.ctrl_connection)

    result = _call_deploy_language_container_deployer_cli(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        itde=itde,
        container_path=None,
        version="0.2.0",
        language_settings=language_settings
    )

    assert result[0][0]


def test_language_container_deployer_cli_with_missing_container_option(
        request, itde):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(itde.ctrl_connection)

    with pytest.raises(Exception) as exc_info:
        _call_deploy_language_container_deployer_cli(
            language_alias="PYTHON3_TE",
            schema=schema_name,
            db_conn=itde.ctrl_connection,
            container_path=None,
            version=None,
            language_settings=language_settings
        )
