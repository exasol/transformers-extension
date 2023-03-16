import textwrap
import pyexasol
import pytest
from click.testing import CliRunner

from exasol_transformers_extension import deploy
from tests.utils.parameters import bucketfs_params, db_params
from tests.utils.revert_language_settings import revert_language_settings
from tests.utils.db_queries import DBQueries
from pathlib import Path


@revert_language_settings
def _call_deploy_language_container_deployer_cli(
        language_alias, schema, db_conn,
        container_path, version, language_settings):
    db_conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
    db_conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

    # call language container deployer
    args_list = [
        "language-container",
        "--bucketfs-name", bucketfs_params.name,
        "--bucketfs-host", bucketfs_params.host,
        "--bucketfs-port", bucketfs_params.port,
        "--bucketfs_use-https", False,
        "--bucketfs-user", bucketfs_params.user,
        "--bucketfs-password", bucketfs_params.password,
        "--bucket", bucketfs_params.bucket,
        "--path-in-bucket", bucketfs_params.path_in_bucket,
        "--container-file", container_path,
        "--version", version,
        "--dsn", db_params.address(),
        "--db-user", db_params.user,
        "--db-pass", db_params.password,
        "--language-alias", language_alias
    ]
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0

    # create a sample UDF using the new language alias
    db_conn_test = pyexasol.connect(
        dsn=db_params.address(),
        user=db_params.user,
        password=db_params.password)
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
        request, pyexasol_connection, language_container):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    result = _call_deploy_language_container_deployer_cli(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        db_conn=pyexasol_connection,
        container_path=Path(language_container["container_path"]),
        version=None,
        language_settings=language_settings
    )

    assert result[0][0]


@pytest.mark.skip(reason="It causes this error:  error:  BucketFS: root path "
                         "'container/language_container'' does not exist in "
                         "bucket 'default' of bucketfs 'bfsdefault'.")
def test_language_container_deployer_cli_by_downloading_container(
        request, pyexasol_connection):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    result = _call_deploy_language_container_deployer_cli(
        language_alias="PYTHON3_TE",
        schema=schema_name,
        db_conn=pyexasol_connection,
        container_path=None,
        version="0.2.0",
        language_settings=language_settings
    )

    assert result[0][0]


def test_language_container_deployer_cli_with_missing_container_option(
        request, pyexasol_connection):
    schema_name = request.node.name
    language_settings = DBQueries.get_language_settings(pyexasol_connection)

    with pytest.raises(Exception) as exc_info:
        _call_deploy_language_container_deployer_cli(
            language_alias="PYTHON3_TE",
            schema=schema_name,
            db_conn=pyexasol_connection,
            container_path=None,
            version=None,
            language_settings=language_settings
        )




