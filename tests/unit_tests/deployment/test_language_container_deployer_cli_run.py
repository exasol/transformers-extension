from pathlib import Path
from unittest.mock import create_autospec, MagicMock
import pytest
from pyexasol import ExaConnection
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from exasol_transformers_extension.deployment.language_container_deployer import (
    LanguageContainerDeployer, LanguageRegLevel)
from exasol_transformers_extension.deployment.language_container_deployer_cli import run_deployer


@pytest.fixture(scope='module')
def mock_pyexasol_conn() -> ExaConnection:
    return create_autospec(ExaConnection)


@pytest.fixture(scope='module')
def mock_bfs_location() -> BucketFSLocation:
    return create_autospec(BucketFSLocation)


@pytest.fixture
def container_deployer(mock_pyexasol_conn, mock_bfs_location) -> LanguageContainerDeployer:
    return LanguageContainerDeployer(pyexasol_connection=mock_pyexasol_conn,
                                     language_alias='alias',
                                     bucketfs_location=mock_bfs_location,
                                     container_file=Path('container_file'))


def test_language_container_deployer_cli_deploy(container_deployer):
    container_deployer.deploy_container = MagicMock()
    run_deployer(container_deployer, True, True)
    container_deployer.deploy_container.assert_called_once()


def test_language_container_deployer_cli_upload(container_deployer):
    container_deployer.upload_container = MagicMock()
    container_deployer.register_container = MagicMock()
    run_deployer(container_deployer, True, False)
    container_deployer.upload_container.assert_called_once()
    container_deployer.register_container.assert_not_called()


def test_language_container_deployer_cli_register(container_deployer):
    container_deployer.upload_container = MagicMock()
    container_deployer.register_container = MagicMock()
    run_deployer(container_deployer, False, True)
    container_deployer.upload_container.assert_not_called()
    container_deployer.register_container.assert_called_once_with(LanguageRegLevel.System)
