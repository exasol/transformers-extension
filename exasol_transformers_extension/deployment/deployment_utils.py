import os
import glob
import logging
import requests
import tempfile
import subprocess
import ssl
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
        response.raise_for_status()
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


def set_websocket_ssl_options(use_ssl_cert_validation: bool, ssl_cert_path: str):
    websocket_sslopt = {
        "cert_reqs": ssl.CERT_REQUIRED,
    }
    if not use_ssl_cert_validation:
        websocket_sslopt["cert_reqs"] = ssl.CERT_NONE

    if ssl_cert_path is not None:
        websocket_sslopt["ca_certs"] = ssl_cert_path
    return websocket_sslopt


@contextmanager
def get_container_file_from_github_release(version):
    with tempfile.TemporaryDirectory() as tmp_dir:
        _download_slc_parts(tmp_dir, version)
        container_file_path = _concatenate_slc_parts(tmp_dir)
        yield container_file_path

