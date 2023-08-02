import contextlib
import os
import subprocess
import textwrap
import time
from pathlib import Path

import pytest
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_script_languages_container_tool.lib.tasks.export.export_info import ExportInfo
from exasol_script_languages_container_tool.lib.tasks.upload.language_definition import LanguageDefinition
from pytest_itde.config import TestConfig

from exasol_transformers_extension.deployment import language_container
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
    container_bucketfs_location = \
        BucketFSFactory().create_bucketfs_location(
            url=f"{itde.bucketfs.url}/{bucketfs_params.bucket}/{bucketfs_params.path_in_bucket};{bucketfs_params.name}",
            user=itde.bucketfs.username,
            pwd=itde.bucketfs.password)
    container_file_path = Path(export_slc.cache_file)
    with open(export_slc.cache_file, "rb") as fileobj:
        container_bucketfs_location.upload_fileobj_to_bucketfs(
            fileobj,
            container_file_path.name)
    with set_language_alias(flavor_path, itde, container_file_path) as language_alias:
        wait_for_language_container_ready(itde, language_alias)
    return container_file_path


@pytest.fixture(scope="session")
def language_alias(itde: TestConfig, flavor_path: Path, upload_slc: Path):
    with set_language_alias(flavor_path, itde, upload_slc) as language_alias:
        yield language_alias


@contextlib.contextmanager
def set_language_alias(flavor_path: Path, itde: TestConfig, container_file_path: Path):
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


def wait_for_language_container_ready(itde: TestConfig, language_alias: str):
    schema = "upload_language_container"
    udf_name = f"wait_{schema}"
    itde.ctrl_connection.execute(f"CREATE SCHEMA If NOT EXISTS {schema}")
    is_ready = False
    wait_time_in_seconds = 180
    for i in range(wait_time_in_seconds):
        time.sleep(1)
        is_ready = is_language_container_ready(itde, language_alias, schema, udf_name)
        if is_ready:
            break
    if not is_ready:
        raise Exception(f"Language container not ready after {wait_time_in_seconds}s.")


def is_language_container_ready(itde: TestConfig, language_alias: str, schema: str, udf_name: str) -> bool:
    try:
        itde.ctrl_connection.execute(textwrap.dedent(f"""
            CREATE OR REPLACE {language_alias} SCALAR SCRIPT {schema}.{udf_name}(i integer) 
            RETURNS INTEGER AS
                def run(ctx):
                    return 1
            / 
            """))
        itde.ctrl_connection.execute(f"SELECT {schema}.{udf_name}(1)")
        return True
    except Exception as e:
        print(e)
        return False
