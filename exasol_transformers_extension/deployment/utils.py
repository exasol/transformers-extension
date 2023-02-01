import os
import glob
import logging
import requests
import tempfile
import subprocess
from pathlib import Path
from getpass import getpass
from contextlib import contextmanager
from jinja2 import Environment, PackageLoader, select_autoescape
from exasol_transformers_extension.deployment import constants


logger = logging.getLogger(__name__)

DB_PASSWORD_ENVIRONMENT_VARIABLE = f"TE_DB_PASSWORD"
BUCKETFS_PASSWORD_ENVIRONMENT_VARIABLE = f"TE_BUCKETFS_PASSWORD"
SLC_PARTS_PREFIX_NAME = 'language_container_part_0'
SLC_FINAL_NAME = "language_container.tar.gz"
N_SLC_PARTS = 2
GH_RELEASE_URL = "https://github.com/exasol/transformers-extension/releases/download"


def get_password(pwd: str, user: str, env_var: str, descr: str) -> str:
    if pwd is None:
        if env_var in os.environ:
            logger.debug(f"Use password from environment variable {env_var}")
            pwd = os.environ[env_var]
        else:
            pwd = getpass(f"{descr} for User {user}")
    return pwd


def load_and_render_statement(template_name, **kwargs) -> str:
    env = Environment(
        loader=PackageLoader(constants.BASE_DIR, constants.TEMPLATES_DIR),
        autoescape=select_autoescape())
    template = env.get_template(template_name)
    statement = template.render(**kwargs)
    return statement


def _download_slc_parts(tmp_dir, version):
    for i in range(N_SLC_PARTS):
        slc_part_name = f"{SLC_PARTS_PREFIX_NAME}" + str(i)
        url = "/".join((GH_RELEASE_URL, version, slc_part_name))
        response = requests.get(url, stream=True)
        with open(Path(tmp_dir, slc_part_name), 'wb') as f:
            f.write(response.content)


def _concatenate_slc_parts(tmp_dir):
    slc_final_path = Path(tmp_dir, SLC_FINAL_NAME)
    slc_parts_path = glob.glob(str(Path(tmp_dir, SLC_PARTS_PREFIX_NAME)) + "*")
    cmd = f"cat {' '.join(slc_parts_path)} > {slc_final_path}"
    subprocess.run(cmd,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT,
                   shell=True)
    return slc_final_path


@contextmanager
def get_container_file_from_github_release(version):
    with tempfile.TemporaryDirectory() as tmp_dir:
        _download_slc_parts(tmp_dir, version)
        container_file_path = _concatenate_slc_parts(tmp_dir)
        yield container_file_path

