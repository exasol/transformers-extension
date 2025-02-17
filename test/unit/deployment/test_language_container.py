from exasol_transformers_extension.deployment.language_container import add_pytorch_to_requirements
from exasol.python_extension_common.deployment.language_container_builder import LanguageContainerBuilder

_dockerfile_template = """
COPY dependencies/requirements.txt /project/requirements.txt
RUN python3.10 -m pip install -r /project/requirements.txt{0}
RUN something_else
"""


def test_add_pytorch_to_requirements():
    source_dockerfile = _dockerfile_template.format('')
    expected_dockerfile = _dockerfile_template.format(
        ' --extra-index-url https://download.pytorch.org/whl/cpu')

    with LanguageContainerBuilder("dummy_container") as container_builder:
        dockerfile_file = 'flavor_base/dependencies/Dockerfile'
        container_builder.write_file(dockerfile_file, source_dockerfile)
        add_pytorch_to_requirements(container_builder)
        assert container_builder.read_file(dockerfile_file) == expected_dockerfile
