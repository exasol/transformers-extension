import logging
import click
from exasol_transformers_extension.deployment.scripts_deployer_cli import \
    scripts_deployer_main
from exasol.python_extension_common.deployment.language_container_deployer_cli \
    import language_container_deployer_main, slc_parameter_formatters, CustomizableParameters
from exasol_transformers_extension.deployment.te_language_container_deployer import TeLanguageContainerDeployer


@click.group()
def main():
    pass


slc_parameter_formatters.set_formatter(CustomizableParameters.container_url,
                                       TeLanguageContainerDeployer.SLC_URL_FORMATTER)
slc_parameter_formatters.set_formatter(CustomizableParameters.container_name,
                                       TeLanguageContainerDeployer.SLC_NAME)

main.add_command(scripts_deployer_main)
main.add_command(language_container_deployer_main)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    main()
