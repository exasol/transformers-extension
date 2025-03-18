"""fixtures for setting up the test db with schema, language alias, udf install"""
from typing import Tuple

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer
from test.utils.parameters import PATH_IN_BUCKET

import pytest
from pyexasol import ExaConnection

BUCKETFS_CONNECTION_NAME = "TEST_TE_BFS_CONNECTION"
SCHEMA_NAME = "TEST_INTEGRATION"
LANGUAGE_ALIAS = "TEST_PYTHON3_TE"


def _deploy_scripts(pyexasol_connection: ExaConnection, install_all_scripts: bool) -> None:
    """installs all existing udfs to test-db"""
    scripts_deployer = ScriptsDeployer(schema=SCHEMA_NAME,
                                       language_alias=LANGUAGE_ALIAS,
                                       install_all_scripts=install_all_scripts,
                                       pyexasol_conn=pyexasol_connection)
    scripts_deployer.deploy_scripts()


@pytest.fixture(scope="session")
def db_schema_name() -> str:
    """get schema name for tests"""
    return SCHEMA_NAME


@pytest.fixture(scope='session')
def language_alias(project_short_tag):
    """get language_alias for TE"""
    return LANGUAGE_ALIAS


@pytest.fixture(scope="session")
def setup_database(backend,
                   pyexasol_connection,
                   bucketfs_connection_factory,
                   deployed_slc) -> Tuple[str, str]:
    """gets a connection to the test-db and installs udfs"""
    # This is a temporary workaround for the problem with slow slc file extraction
    # at a SaaS database. To be removed when a proper completion check is in place.
    if backend == 'saas':
        import time
        time.sleep(30)

    bucketfs_connection_factory(BUCKETFS_CONNECTION_NAME, PATH_IN_BUCKET)
    _deploy_scripts(pyexasol_connection, install_all_scripts=True)
    return BUCKETFS_CONNECTION_NAME, SCHEMA_NAME


@pytest.fixture()
def db_conn(setup_database, pyexasol_connection) -> ExaConnection:
    """
    Per-test fixture that returns the same session-wide pyexasol connection,
    but makes sure the default schema is open.
    """
    pyexasol_connection.execute(f"OPEN SCHEMA {SCHEMA_NAME};")
    return pyexasol_connection
