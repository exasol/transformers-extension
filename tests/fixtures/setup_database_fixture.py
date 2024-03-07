from typing import Tuple
from urllib.parse import urlparse

import pytest
from pyexasol import ExaConnection
from pytest_itde import config

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.parameters import bucketfs_params
from tests.fixtures.language_container_fixture import language_alias

bucketfs_connection_name = "TEST_TE_BFS_CONNECTION"
schema_name = "TEST_INTEGRATION"


def _create_schema(pyexasol_connection: ExaConnection) -> None:
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")
    pyexasol_connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")


def _deploy_scripts(pyexasol_connection: ExaConnection, language_alias: str) -> None:
    scripts_deployer = ScriptsDeployer(schema=schema_name,
                                       language_alias=language_alias,
                                       pyexasol_conn=pyexasol_connection)
    scripts_deployer.deploy_scripts()


def _create_bucketfs_connection(bucketfs_config: config.BucketFs,
                                pyexasol_connection: ExaConnection) -> None:
    parsed_url = urlparse(bucketfs_config.url)
    host = parsed_url.netloc.split(":")[0]
    address = f"{parsed_url.scheme}://{host}:{bucketfs_params.real_port}/{bucketfs_params.bucket}/" \
              f"{bucketfs_params.path_in_bucket};{bucketfs_params.name}"
    query = f"CREATE OR REPLACE  CONNECTION {bucketfs_connection_name} " \
            f"TO '{address}' " \
            f"USER '{bucketfs_config.username}' " \
            f"IDENTIFIED BY '{bucketfs_config.password}'"
    pyexasol_connection.execute(query)


@pytest.fixture(scope="module")
def setup_database(bucketfs_config: config.BucketFs,
                   pyexasol_connection: ExaConnection,
                   language_alias: str) -> Tuple[str, str]:
    _create_schema(pyexasol_connection)
    _deploy_scripts(pyexasol_connection, language_alias)
    _create_bucketfs_connection(bucketfs_config, pyexasol_connection)
    return bucketfs_connection_name, schema_name
