from typing import Tuple
from urllib.parse import urlparse

import pytest
from pytest_itde.config import TestConfig

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.parameters import bucketfs_params

bucketfs_connection_name = "TEST_TE_BFS_CONNECTION"
schema_name = "TEST_INTEGRATION"
language_alias = "PYTHON3_TE"


def _create_schema(db_conn) -> None:
    db_conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
    db_conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")


def _deploy_scripts(itde: TestConfig) -> None:
    ScriptsDeployer.run(
        dsn=f"{itde.db.host}:{itde.db.port}",
        user=itde.db.username,
        password=itde.db.password,
        schema=schema_name,
        language_alias=language_alias
    )


def _create_bucketfs_connection(itde: TestConfig) -> None:
    parsed_url = urlparse(itde.bucketfs.url)
    host = parsed_url.netloc.split(":")[0]
    address = f"{parsed_url.scheme}://{host}:{bucketfs_params.real_port}/{bucketfs_params.bucket}/" \
              f"{bucketfs_params.path_in_bucket};{bucketfs_params.name}"
    query = f"CREATE OR REPLACE  CONNECTION {bucketfs_connection_name} " \
            f"TO '{address}' " \
            f"USER '{itde.bucketfs.username}' " \
            f"IDENTIFIED BY '{itde.bucketfs.password}'"
    itde.ctrl_connection.execute(query)


@pytest.fixture(scope="module")
def setup_database(itde: TestConfig) -> Tuple[str, str]:
    _create_schema(itde.ctrl_connection)
    _deploy_scripts(itde)
    _create_bucketfs_connection(itde)
    return bucketfs_connection_name, schema_name
