
import logging
import requests
import tempfile
import ssl
from pathlib import Path
from contextlib import contextmanager
from jinja2 import Environment, PackageLoader, select_autoescape, ChoiceLoader
from exasol_transformers_extension.deployment import constants, work_with_spans_constants, work_without_spans_constants


logger = logging.getLogger(__name__)


DB_PASSWORD_ENVIRONMENT_VARIABLE = f"TE_DB_PASSWORD" #todo should these move to constants?
BUCKETFS_PASSWORD_ENVIRONMENT_VARIABLE = f"TE_BUCKETFS_PASSWORD"


def load_and_render_statement(template_name, work_with_spans, install_all_scripts, **kwargs) -> str:
    package_loaders = [PackageLoader(constants.BASE_DIR, constants.TEMPLATES_DIR)]
    if work_with_spans or install_all_scripts:
        package_loaders.append(PackageLoader(work_with_spans_constants.BASE_DIR, work_with_spans_constants.TEMPLATES_DIR))
    if install_all_scripts or not work_with_spans:
        package_loaders.append(PackageLoader(work_without_spans_constants.BASE_DIR, work_without_spans_constants.TEMPLATES_DIR))

    env = Environment(
        loader=ChoiceLoader(package_loaders),
        autoescape=select_autoescape())

    template = env.get_template(template_name)
    statement = template.render(**kwargs)
    return statement


def get_websocket_ssl_options(use_ssl_cert_validation: bool, ssl_cert_path: str):
    websocket_sslopt = {
        "cert_reqs": ssl.CERT_REQUIRED,
    }
    if not use_ssl_cert_validation:
        websocket_sslopt["cert_reqs"] = ssl.CERT_NONE

    if ssl_cert_path is not None:
        websocket_sslopt["ca_certs"] = ssl_cert_path
    return websocket_sslopt
