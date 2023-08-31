import os
import click
from exasol_transformers_extension.deployment import deployment_utils as utils
from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer


@click.command(name="scripts")
@click.option('--dsn', type=str, required=True)
@click.option('--db-user', type=str, required=True)
@click.option('--db-pass', prompt='db password', hide_input=True,
              default=lambda: os.environ.get(
                  utils.DB_PASSWORD_ENVIRONMENT_VARIABLE, ""))
@click.option('--schema', type=str, required=True)
@click.option('--language-alias', type=str, default="PYTHON3_TE")
@click.option('--ssl_cert_path', type=str, default="")
@click.option('--use_ssl_cert_validation', type=bool, default=True)
def scripts_deployer_main(
        dsn: str, db_user: str, db_pass: str, schema: str, language_alias: str,
        ssl_cert_path: str, use_ssl_cert_validation: bool):

    ScriptsDeployer.run(
        dsn=dsn,
        user=db_user,
        password=db_pass,
        schema=schema,
        language_alias=language_alias,
        ssl_cert_path=ssl_cert_path,
        use_ssl_cert_validation=use_ssl_cert_validation
    )


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    scripts_deployer_main()
