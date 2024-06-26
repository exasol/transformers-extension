from typing import Tuple
import json

from urllib.parse import urlparse
import pytest
from pyexasol import ExaConnection
from pytest_itde import config
import exasol.bucketfs as bfs

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
    def to_json_str(**kwargs) -> str:
        filtered_kwargs = {k: v for k,v in kwargs.items() if v is not None}
        return json.dumps(filtered_kwargs)

    parsed_url = urlparse(bucketfs_config.url)
    host = parsed_url.netloc.split(":")[0]
    url = f"{parsed_url.scheme}://{host}:{bucketfs_params.real_port}"
    conn_to = to_json_str(backend=bfs.path.StorageBackend.onprem.name,
                          url=url,
                          service_name=bucketfs_params.name,
                          bucket_name=bucketfs_params.bucket,
                          path=bucketfs_params.path_in_bucket,
                          verify=False)
    conn_user = to_json_str(username=bucketfs_config.username)
    conn_password = to_json_str(password=bucketfs_config.password)

    query = f"CREATE OR REPLACE  CONNECTION {bucketfs_connection_name} " \
            f"TO '{conn_to}' " \
            f"USER '{conn_user}' " \
            f"IDENTIFIED BY '{conn_password}'"
    pyexasol_connection.execute(query)


@pytest.fixture(scope="module")
def setup_database(bucketfs_config: config.BucketFs,
                   pyexasol_connection: ExaConnection,
                   language_alias: str) -> Tuple[str, str]:
    _create_schema(pyexasol_connection)
    _deploy_scripts(pyexasol_connection, language_alias)
    _create_bucketfs_connection(bucketfs_config, pyexasol_connection)
    return bucketfs_connection_name, schema_name
