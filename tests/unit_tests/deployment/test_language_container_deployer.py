import pytest
from pathlib import Path
from unittest.mock import create_autospec, MagicMock
from pyexasol import ExaConnection
import exasol.bucketfs as bfs

from exasol_transformers_extension.deployment.te_language_container_deployer import TeLanguageContainerDeployer
from exasol.python_extension_common.deployment.language_container_deployer import LanguageActivationLevel


@pytest.fixture
def te_container_deployer() -> TeLanguageContainerDeployer:
    deployer = TeLanguageContainerDeployer(pyexasol_connection=create_autospec(ExaConnection),
                                           language_alias='PYTHON3_TEST',
                                           bucketfs_path=create_autospec(bfs.path.PathLike))
    deployer.upload_container = MagicMock()
    deployer.activate_container = MagicMock()
    return deployer


def test_te_language_container_deployer(te_container_deployer: TeLanguageContainerDeployer):
    file_name = "te_container.tar.gz"
    file_path = Path(file_name)
    te_container_deployer.run(container_file=file_path,
                              bucket_file_path=file_name,
                              alter_system=True,
                              allow_override=True)
    te_container_deployer.upload_container.assert_called_once_with(file_path, file_name)
    te_container_deployer.activate_container.assert_called_once_with(
        file_name, LanguageActivationLevel.System, True)
