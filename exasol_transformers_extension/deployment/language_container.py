import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from exasol_integration_test_docker_environment.lib.docker.images.image_info import ImageInfo
from exasol_script_languages_container_tool.lib import api
from exasol_script_languages_container_tool.lib.tasks.export.export_containers import ExportContainerResult


def find_file_or_folder_backwards(name: str) -> Path:
    current_path = Path(__file__).parent
    result_path = None
    while current_path != current_path.root:
        result_path = Path(current_path, name)
        if result_path.exists():
            break
        current_path = current_path.parent
    if result_path is not None and result_path.exists():
        return result_path
    else:
        raise RuntimeError(f"Could not find {name} when searching backwards from {Path(__file__).parent}")


CONTAINER_NAME = "exasol_transformers_extension_container"


def find_flavor_path() -> Path:
    language_container_path = find_file_or_folder_backwards("language_container")
    flavor_path = language_container_path / CONTAINER_NAME
    return flavor_path


def build_language_container(flavor_path: Path) -> Dict[str, ImageInfo]:
    image_infos = api.build(flavor_path=(str(flavor_path),), goal=("release",))
    return image_infos


def export(flavor_path: Path,
           export_path: Optional[Path] = None) -> ExportContainerResult:
    if export_path is not None:
        export_path = str(export_path)
    export_result = api.export(flavor_path=(str(flavor_path),), export_path=export_path)
    return export_result


def upload(
        flavor_path: Path,
        bucketfs_name: str,
        bucket_name: str,
        database_host: str,
        bucketfs_port: int,
        user: str,
        password: str,
        path_in_bucket: str,
        release_name: str
):
    api.upload(
        flavor_path=(str(flavor_path),),
        bucketfs_name=bucketfs_name,
        bucket_name=bucket_name,
        bucketfs_port=bucketfs_port,
        database_host=database_host,
        bucketfs_username=user,
        bucketfs_password=password,
        path_in_bucket=path_in_bucket,
        release_name=release_name
    )


def prepare_flavor(flavor_path: Path):
    flavor_base_path = flavor_path / "flavor_base"
    add_requirements_to_flavor(flavor_base_path)
    add_wheel_to_flavor(flavor_base_path)


def find_project_directory():
    project_directory = find_file_or_folder_backwards("pyproject.toml").parent
    return project_directory


def add_wheel_to_flavor(flavor_base_path):
    project_directory = find_project_directory()
    subprocess.call(["poetry", "build"], cwd=project_directory)
    dist_path = project_directory / "dist"
    wheels = list(dist_path.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Did not find exactly one wheel file in dist directory {dist_path}. "
                           f"Found the following wheels: {wheels}")
    wheel = wheels[0]
    wheel_target = flavor_base_path / "release" / "dist"
    wheel_target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wheel, wheel_target / wheel.name)


def add_requirements_to_flavor(flavor_base_path: Path):
    project_directory = find_project_directory()
    requirements_bytes = subprocess.check_output(["poetry", "export", "--without-hashes", "--without-urls"],
                                                 cwd=project_directory)
    requirements = requirements_bytes.decode("UTF-8")
    requirements_without_cuda = "\n".join(line
                                          for line in requirements.splitlines()
                                          if not line.startswith("nvidia"))
    requirements_file = flavor_base_path / "dependencies" / "requirements.txt"
    requirements_file.write_text(requirements_without_cuda)
