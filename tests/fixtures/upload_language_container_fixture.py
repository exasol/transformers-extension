import subprocess
import textwrap
import time
from urllib.parse import urlparse

import pytest
from exasol_script_languages_container_tool.lib.tasks.upload.language_definition import LanguageDefinition
from pytest_itde.config import TestConfig

from exasol_transformers_extension.deployment import language_container
from tests.utils.parameters import bucketfs_params


@pytest.fixture(scope="session")
def upload_language_container(itde: TestConfig) -> str:
    # Remove image and build output to reduce the disk usage in CI.
    # We currently, use Github Actions as the CI and its disk is limited to 14 GB.
    # TODO: This code can be removed if we moved to a CI with larger disks.
    rm_docker_image = """docker images -a | grep 'transformers' | awk '{print $3}' | xargs docker rmi"""
    subprocess.run(rm_docker_image, shell=True)

    flavor_path = language_container.find_flavor_path()
    parsed_url = urlparse(itde.bucketfs.url)
    language_container.prepare_flavor(flavor_path=flavor_path)
    export_result = language_container.export(flavor_path=flavor_path)
    release_name = "test"
    language_container.upload(
        flavor_path=flavor_path,
        bucketfs_name=bucketfs_params.name,
        bucket_name=bucketfs_params.bucket,
        database_host=parsed_url.hostname,
        bucketfs_port=parsed_url.port,
        user=itde.bucketfs.username,
        password=itde.bucketfs.password,
        path_in_bucket=bucketfs_params.path_in_bucket,
        release_name=release_name
    )
    export_info = export_result.export_infos[str(flavor_path)]["release"]
    complete_release_name = f"""{export_info.name}-{export_info.release_goal}-{release_name}"""
    language_definition = LanguageDefinition(
        flavor_path=str(flavor_path),
        bucketfs_name=bucketfs_params.name,
        bucket_name=bucketfs_params.bucket,
        add_missing_builtin=False,
        path_in_bucket=bucketfs_params.path_in_bucket,
        release_name=complete_release_name
    )
    result = itde.ctrl_connection.execute(
        f"""SELECT "SYSTEM_VALUE" FROM SYS.EXA_PARAMETERS WHERE
         PARAMETER_NAME='SCRIPT_LANGUAGES'"""
    ).fetchall()
    original_alter_system = result[0][0]
    itde.ctrl_connection.execute(language_definition.generate_alter_system())
    itde.ctrl_connection.execute(language_definition.generate_alter_session())
    language_alias = language_definition.generate_definition().split("=")[0]
    wait_for_language_container_ready(itde, language_alias)
    yield language_alias
    itde.ctrl_connection.execute(f"ALTER SYSTEM SET SCRIPT_LANGUAGES='{original_alter_system}'")


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
