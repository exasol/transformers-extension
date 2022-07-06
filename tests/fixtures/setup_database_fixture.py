import pytest
from typing import Tuple
from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.parameters import db_params, bucketfs_params


bucketfs_connection_name = "TEST_TE_BFS_CONNECTION"
schema_name = "TEST_INTEGRATION"
language_alias = "PYTHON3_TE"


def _create_schema(db_conn) -> None:
    db_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
    db_conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")


def _deploy_scripts() -> None:
    ScriptsDeployer.run(
        dsn=db_params.address(),
        user=db_params.user,
        password=db_params.password,
        schema=schema_name,
        language_alias=language_alias
    )


def _create_bucketfs_connection(db_conn) -> None:
    query = f"CREATE OR REPLACE  CONNECTION {bucketfs_connection_name} " \
            f"TO '{bucketfs_params.address(bucketfs_params.real_port)}' " \
            f"USER '{bucketfs_params.user}' " \
            f"IDENTIFIED BY '{bucketfs_params.password}'"
    db_conn.execute(query)


@pytest.fixture(scope="module")
def setup_database(pyexasol_connection) -> Tuple[str, str]:
    _create_schema(pyexasol_connection)
    _deploy_scripts()
    _create_bucketfs_connection(pyexasol_connection)
    return bucketfs_connection_name, schema_name
