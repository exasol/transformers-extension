import contextlib
import os
import subprocess
from pathlib import Path

import pytest
from exasol_script_languages_container_tool.lib.tasks.export.export_info import ExportInfo
from exasol_script_languages_container_tool.lib.tasks.upload.language_definition import LanguageDefinition
from pytest_itde.config import TestConfig
from exasol.python_extension_common.deployment.language_container_validator import (
    wait_language_container, temp_schema)

from exasol_transformers_extension.deployment import language_container
from exasol_transformers_extension.utils.bucketfs_operations import create_bucketfs_location
from tests.utils.parameters import bucketfs_params
from tests.utils.revert_language_settings import revert_language_settings


@pytest.fixture(scope="session")
def flavor_path() -> Path:
    return language_container.find_flavor_path()


@pytest.fixture(scope="session")
def export_slc(flavor_path: Path) -> ExportInfo:
    language_container.prepare_flavor(flavor_path=flavor_path)
    export_result = language_container.export(flavor_path=flavor_path)
    export_info = export_result.export_infos[str(flavor_path)]["release"]
    return export_info


@pytest.fixture(scope="session")
def upload_slc(itde: TestConfig, flavor_path: Path, export_slc: ExportInfo) -> Path:
    cleanup_images()
    container_bucketfs_location = create_bucketfs_location(
        path_in_bucket=bucketfs_params.path_in_bucket,
        bucketfs_name=bucketfs_params.name,
        bucketfs_url=itde.bucketfs.url,
        bucketfs_user=itde.bucketfs.username,
        bucketfs_password=itde.bucketfs.password,
        bucket=bucketfs_params.bucket,
        use_ssl_cert_validation=False)
    container_file_path = Path(export_slc.cache_file)
    container_file_bfs_path = container_bucketfs_location / container_file_path.name

    with open(export_slc.cache_file, "rb") as fileobj:
        container_file_bfs_path.write(fileobj)

    with set_language_alias(flavor_path, itde, container_file_path) as language_alias:
        with temp_schema(itde.ctrl_connection) as schema:
            wait_language_container(itde.ctrl_connection, language_alias, schema)
    return container_file_path


@pytest.fixture(scope="session")
def language_alias(itde: TestConfig, flavor_path: Path, upload_slc: Path) -> str:
    with set_language_alias(flavor_path, itde, upload_slc) as language_alias:
        yield language_alias


@contextlib.contextmanager
def set_language_alias(flavor_path: Path, itde: TestConfig, container_file_path: Path) -> str:
    release_name = container_file_path.with_suffix('').with_suffix('').name
    language_definition = LanguageDefinition(
        flavor_path=str(flavor_path),
        bucketfs_name=bucketfs_params.name,
        bucket_name=bucketfs_params.bucket,
        add_missing_builtin=False,
        path_in_bucket=bucketfs_params.path_in_bucket,
        release_name=release_name
    )
    with revert_language_settings(itde.ctrl_connection):
        language_alias = language_definition.generate_definition().split("=")[0]
        itde.ctrl_connection.execute(language_definition.generate_alter_session())
        itde.ctrl_connection.execute(language_definition.generate_alter_system())
        yield language_alias


def cleanup_images():
    if "GITHUB_ACTIONS" in os.environ:
        # Remove image and build output to reduce the disk usage in CI.
        # We currently use Github Actions as the CI and its disk is limited to 14 GB.
        # TODO: This code can be removed when we move to a CI with larger disks.
        rm_docker_image = """docker images -a | grep 'transformers' | awk '{print $3}' | xargs docker rmi"""
        subprocess.run(rm_docker_image, shell=True)
