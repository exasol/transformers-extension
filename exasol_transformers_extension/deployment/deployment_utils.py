
import logging
import ssl

from jinja2 import Environment, PackageLoader, select_autoescape, ChoiceLoader
from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.deployment.work_with_spans_constants import work_with_spans_constants
from exasol_transformers_extension.deployment.work_without_spans_constants import work_without_spans_constants


logger = logging.getLogger(__name__)

def load_and_render_statement(template_name, work_with_spans, install_all_scripts, **kwargs) -> str:
    package_loaders = [PackageLoader(constants.base_dir, constants.templates_dir)]
    if work_with_spans or install_all_scripts:
        package_loaders.append(PackageLoader(work_with_spans_constants.base_dir, work_with_spans_constants.templates_dir))
    if install_all_scripts or not work_with_spans:
        package_loaders.append(PackageLoader(work_without_spans_constants.base_dir, work_without_spans_constants.templates_dir))

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
