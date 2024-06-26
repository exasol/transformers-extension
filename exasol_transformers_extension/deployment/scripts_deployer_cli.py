from __future__ import annotations

import click
from exasol.python_extension_common.deployment.language_container_deployer_cli import (
    SecretParams, SECRET_DISPLAY, secret_callback)

from exasol_transformers_extension.deployment.scripts_deployer import \
    ScriptsDeployer


@click.command(name="scripts")
@click.option('--dsn', type=str)
@click.option('--db-user', type=str)
@click.option(f'--{SecretParams.DB_PASSWORD.value}', type=str,
              prompt='DB password', prompt_required=False,
              hide_input=True, default=SECRET_DISPLAY, callback=secret_callback)
@click.option('--saas-url', type=str,
              default='https://cloud.exasol.com')
@click.option(f'--{SecretParams.SAAS_ACCOUNT_ID.value}', type=str,
              prompt='SaaS account id', prompt_required=False,
              hide_input=True, default=SECRET_DISPLAY, callback=secret_callback)
@click.option(f'--{SecretParams.SAAS_DATABASE_ID.value}', type=str,
              prompt='SaaS database id', prompt_required=False,
              hide_input=True, default=SECRET_DISPLAY, callback=secret_callback)
@click.option('--saas-database-name', type=str)
@click.option(f'--{SecretParams.SAAS_TOKEN.value}', type=str,
              prompt='SaaS token', prompt_required=False,
              hide_input=True, default=SECRET_DISPLAY, callback=secret_callback)
@click.option('--ssl-cert-path', type=str, default="")
@click.option('--ssl-client-cert-path', type=str, default="")
@click.option('--ssl-client-private-key', type=str, default="")
@click.option('--use-ssl-cert-validation/--no-use-ssl-cert-validation', type=bool, default=True)
@click.option('--schema', type=str, required=True, help="schema name")
@click.option('--language-alias', type=str, default="PYTHON3_TE")
def scripts_deployer_main(
        dsn: str | None,
        db_user: str | None,
        db_pass: str | None,
        saas_url: str | None,
        saas_account_id: str | None,
        saas_database_id: str | None,
        saas_database_name: str | None,
        saas_token: str | None,
        ssl_cert_path: str,
        ssl_client_cert_path: str,
        ssl_client_private_key: str,
        use_ssl_cert_validation: bool,
        schema: str,
        language_alias: str):

    ScriptsDeployer.run(
        language_alias=language_alias,
        schema=schema,
        dsn=dsn,
        db_user=db_user,
        db_pass=db_pass,
        saas_url=saas_url,
        saas_account_id=saas_account_id,
        saas_database_id=saas_database_id,
        saas_database_name=saas_database_name,
        saas_token=saas_token,
        use_ssl_cert_validation=use_ssl_cert_validation,
        ssl_trusted_ca=ssl_cert_path,
        ssl_client_certificate=ssl_client_cert_path,
        ssl_private_key=ssl_client_private_key,
    )


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    scripts_deployer_main()
