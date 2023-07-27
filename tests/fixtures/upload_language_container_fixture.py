import subprocess

import pytest
from pathlib import Path
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from pytest_itde.config import TestConfig

from tests.utils.parameters import bucketfs_params


@pytest.fixture(scope="session")
def upload_language_container(itde: TestConfig, language_container) -> str:
    bucket_fs_factory = BucketFSFactory()
    container_bucketfs_location = \
        bucket_fs_factory.create_bucketfs_location(
            url=f"{itde.bucketfs.url}/{bucketfs_params.bucket}/{bucketfs_params.path_in_bucket};{bucketfs_params.name}",
            user=itde.bucketfs.username,
            pwd=itde.bucketfs.password)
    container_path = Path(language_container["container_path"])
    alter_session = language_container["alter_session"]
    language_alias = alter_session.split("=")[0]
    with open(container_path, "rb") as container_file:
        container_bucketfs_location.upload_fileobj_to_bucketfs(
            container_file,
            "exasol_transformers_extension_container.tar.gz")

    # Remove image and build output to reduce the disk usage in CI.
    # We currently, use Github Actions as the CI and its disk is limited to 14 GB.
    # TODO: This code can be removed if we moved to a CI with larger disks.
    rm_docker_image = """docker images -a | grep 'transformers' | awk '{print $3}' | xargs docker rmi"""
    subprocess.run(rm_docker_image, shell=True)

    result = itde.ctrl_connection.execute(
        f"""SELECT "SYSTEM_VALUE" FROM SYS.EXA_PARAMETERS WHERE 
        PARAMETER_NAME='SCRIPT_LANGUAGES'""").fetchall()
    original_alter_system = result[0][0]
    itde.ctrl_connection.execute(
        f"ALTER SESSION SET SCRIPT_LANGUAGES='{alter_session}'")
    itde.ctrl_connection.execute(
        f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{alter_session}'")

    yield language_alias
    itde.ctrl_connection.execute(
        f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{original_alter_system}'")
