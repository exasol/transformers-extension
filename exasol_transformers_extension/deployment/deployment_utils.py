
import logging
import requests
import tempfile
import ssl
from pathlib import Path
from contextlib import contextmanager
from jinja2 import Environment, PackageLoader, select_autoescape
from exasol_transformers_extension.deployment import constants


logger = logging.getLogger(__name__)


DB_PASSWORD_ENVIRONMENT_VARIABLE = f"TE_DB_PASSWORD"
BUCKETFS_PASSWORD_ENVIRONMENT_VARIABLE = f"TE_BUCKETFS_PASSWORD"
SLC_NAME = "exasol_transformers_extension_container_release.tar.gz"
GH_RELEASE_URL = "https://github.com/exasol/transformers-extension/releases/download"


def load_and_render_statement(template_name, **kwargs) -> str:
    env = Environment(
        loader=PackageLoader(constants.BASE_DIR, constants.TEMPLATES_DIR),
        autoescape=select_autoescape())
    template = env.get_template(template_name)
    statement = template.render(**kwargs)
    return statement


def _download_slc(tmp_dir: Path, version: str) -> Path:
    url = "/".join((GH_RELEASE_URL, version, SLC_NAME))
    response = requests.get(url, stream=True)
    response.raise_for_status()
    slc_path = Path(tmp_dir, SLC_NAME)
    with open(slc_path, 'wb') as f:
        f.write(response.content)
    return slc_path


def get_websocket_ssl_options(use_ssl_cert_validation: bool, ssl_cert_path: str):
    websocket_sslopt = {
        "cert_reqs": ssl.CERT_REQUIRED,
    }
    if not use_ssl_cert_validation:
        websocket_sslopt["cert_reqs"] = ssl.CERT_NONE

    if ssl_cert_path is not None:
        websocket_sslopt["ca_certs"] = ssl_cert_path
    return websocket_sslopt


@contextmanager
def get_container_file_from_github_release(version: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        container_file_path = _download_slc(tmp_dir, version)
        yield container_file_path

