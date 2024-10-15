import logging
from jinja2 import Environment, PackageLoader, select_autoescape
from exasol_transformers_extension.deployment import constants


logger = logging.getLogger(__name__)


def load_and_render_statement(template_name, **kwargs) -> str:
    env = Environment(
        loader=PackageLoader(constants.BASE_DIR, constants.TEMPLATES_DIR),
        autoescape=select_autoescape())
    template = env.get_template(template_name)
    statement = template.render(**kwargs)
    return statement
