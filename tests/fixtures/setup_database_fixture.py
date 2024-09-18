from typing import Tuple
import pytest
from pyexasol import ExaConnection

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from tests.utils.parameters import PATH_IN_BUCKET
from tests.fixtures.language_container_fixture_constants import LANGUAGE_ALIAS

BUCKETFS_CONNECTION_NAME = "TEST_TE_BFS_CONNECTION"
SCHEMA_NAME = "TEST_INTEGRATION"


def _deploy_scripts(pyexasol_connection: ExaConnection) -> None:
    scripts_deployer = ScriptsDeployer(schema=SCHEMA_NAME,
                                       language_alias=LANGUAGE_ALIAS,
                                       pyexasol_conn=pyexasol_connection)
    scripts_deployer.deploy_scripts()


@pytest.fixture(scope="session")
def db_schema_name() -> str:
    return SCHEMA_NAME


@pytest.fixture(scope="session")
def setup_database(pyexasol_connection,
                   bucketfs_connection_factory,
                   upload_slc) -> Tuple[str, str]:
    bucketfs_connection_factory(BUCKETFS_CONNECTION_NAME, PATH_IN_BUCKET)
    _deploy_scripts(pyexasol_connection)
    return BUCKETFS_CONNECTION_NAME, SCHEMA_NAME


@pytest.fixture()
def db_conn(setup_database, pyexasol_connection) -> ExaConnection:
    """
    Per-test fixture that returns the same session-wide pyexasol connection,
    but makes sure the default schema is open.
    """
    pyexasol_connection.execute(f"OPEN SCHEMA {SCHEMA_NAME};")
    return pyexasol_connection
