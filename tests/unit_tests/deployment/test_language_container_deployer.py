#########################################################
# To be migrated to the script-languages-container-tool #
#########################################################
from pathlib import Path, PurePosixPath
from unittest.mock import create_autospec, MagicMock, patch

import pytest
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from pyexasol import ExaConnection

from exasol_transformers_extension.deployment.language_container_deployer import (
    LanguageContainerDeployer, LanguageActivationLevel)


@pytest.fixture(scope='module')
def container_file_name() -> str:
    return 'container_xyz.tag.gz'


@pytest.fixture(scope='module')
def container_file_path(container_file_name) -> Path:
    return Path(container_file_name)


@pytest.fixture(scope='module')
def language_alias() -> str:
    return 'PYTHON3_TEST'


@pytest.fixture(scope='module')
def container_bfs_path(container_file_name) -> str:
    return f'bfsdefault/default/container/{container_file_name[:-7]}'


@pytest.fixture(scope='module')
def mock_pyexasol_conn() -> ExaConnection:
    return create_autospec(ExaConnection)


@pytest.fixture(scope='module')
def mock_bfs_location(container_bfs_path) -> BucketFSLocation:
    mock_loc = create_autospec(BucketFSLocation)
    mock_loc.generate_bucket_udf_path.return_value = PurePosixPath(f'/buckets/{container_bfs_path}')
    return mock_loc


@pytest.fixture
def container_deployer(mock_pyexasol_conn, mock_bfs_location, language_alias) -> LanguageContainerDeployer:
    deployer = LanguageContainerDeployer(pyexasol_connection=mock_pyexasol_conn,
                                         language_alias=language_alias,
                                         bucketfs_location=mock_bfs_location)

    deployer.upload_container = MagicMock()
    deployer.activate_container = MagicMock()
    return deployer


def test_slc_deployer_deploy(container_deployer, container_file_name, container_file_path):
    container_deployer.run(container_file=container_file_path, bucket_file_path=container_file_name, alter_system=True,
                           allow_override=True)
    container_deployer.upload_container.assert_called_once_with(container_file_path, container_file_name)
    container_deployer.activate_container.assert_called_once_with(container_file_name, LanguageActivationLevel.System,
                                                                  True)


def test_slc_deployer_upload(container_deployer, container_file_name, container_file_path):
    container_deployer.run(container_file=container_file_path, alter_system=False)
    container_deployer.upload_container.assert_called_once_with(container_file_path, container_file_name)
    container_deployer.activate_container.assert_not_called()


def test_slc_deployer_activate(container_deployer, container_file_name, container_file_path):
    container_deployer.run(bucket_file_path=container_file_name, alter_system=True, allow_override=True)
    container_deployer.upload_container.assert_not_called()
    container_deployer.activate_container.assert_called_once_with(container_file_name, LanguageActivationLevel.System,
                                                                  True)


@patch('exasol_transformers_extension.deployment.language_container_deployer.get_language_settings')
def test_slc_deployer_generate_activation_command(mock_lang_settings, container_deployer, language_alias,
                                                  container_file_name, container_bfs_path):
    mock_lang_settings.return_value = 'R=builtin_r JAVA=builtin_java PYTHON3=builtin_python3'

    alter_type = LanguageActivationLevel.Session
    expected_command = f"ALTER {alter_type.value.upper()} SET SCRIPT_LANGUAGES='" \
                       "R=builtin_r JAVA=builtin_java PYTHON3=builtin_python3 " \
                       f"{language_alias}=localzmq+protobuf:///{container_bfs_path}?" \
                       f"lang=python#/buckets/{container_bfs_path}/exaudf/exaudfclient_py3';"

    command = container_deployer.generate_activation_command(container_file_name, alter_type)
    assert command == expected_command


@patch('exasol_transformers_extension.deployment.language_container_deployer.get_language_settings')
def test_slc_deployer_generate_activation_command_override(mock_lang_settings, container_deployer, language_alias,
                                                           container_file_name, container_bfs_path):
    current_bfs_path = 'bfsdefault/default/container_abc'
    mock_lang_settings.return_value = \
        'R=builtin_r JAVA=builtin_java PYTHON3=builtin_python3 ' \
        f'{language_alias}=localzmq+protobuf:///{current_bfs_path}?' \
        f'lang=python#/buckets/{current_bfs_path}/exaudf/exaudfclient_py3'

    alter_type = LanguageActivationLevel.Session
    expected_command = f"ALTER {alter_type.value.upper()} SET SCRIPT_LANGUAGES='" \
                       "R=builtin_r JAVA=builtin_java PYTHON3=builtin_python3 " \
                       f"{language_alias}=localzmq+protobuf:///{container_bfs_path}?" \
                       f"lang=python#/buckets/{container_bfs_path}/exaudf/exaudfclient_py3';"

    command = container_deployer.generate_activation_command(container_file_name, alter_type, allow_override=True)
    assert command == expected_command


@patch('exasol_transformers_extension.deployment.language_container_deployer.get_language_settings')
def test_slc_deployer_generate_activation_command_failure(mock_lang_settings, container_deployer, language_alias,
                                                          container_file_name):
    current_bfs_path = 'bfsdefault/default/container_abc'
    mock_lang_settings.return_value = \
        'R=builtin_r JAVA=builtin_java PYTHON3=builtin_python3 ' \
        f'{language_alias}=localzmq+protobuf:///{current_bfs_path}?' \
        f'lang=python#/buckets/{current_bfs_path}/exaudf/exaudfclient_py3'

    with pytest.raises(RuntimeError):
        container_deployer.generate_activation_command(container_file_name, LanguageActivationLevel.Session,
                                                       allow_override=False)


def test_slc_deployer_get_language_definition(container_deployer, language_alias,
                                              container_file_name, container_bfs_path):
    expected_command = f"{language_alias}=localzmq+protobuf:///{container_bfs_path}?" \
                       f"lang=python#/buckets/{container_bfs_path}/exaudf/exaudfclient_py3"

    command = container_deployer.get_language_definition(container_file_name)
    assert command == expected_command
