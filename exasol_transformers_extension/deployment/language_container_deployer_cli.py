import os
import click
from pathlib import Path
from textwrap import dedent
from exasol_transformers_extension.deployment import deployment_utils as utils
from exasol_transformers_extension.deployment.language_container_deployer import \
    LanguageContainerDeployer, LanguageActivationLevel


def run_deployer(deployer, upload_container: bool = True,
                 alter_system: bool = True,
                 allow_override: bool = False) -> None:
    if upload_container and alter_system:
        deployer.deploy_container(allow_override)
    elif upload_container:
        deployer.upload_container()
    elif alter_system:
        deployer.activate_container(LanguageActivationLevel.System, allow_override)

    if not alter_system:
        message = dedent(f"""
        In SQL, you can activate the SLC of the Transformer Extension
        by using the following statements:

        To activate the SLC only for the current session:
        {deployer.generate_activation_command(LanguageActivationLevel.Session, True)}

        To activate the SLC on the system:
        {deployer.generate_activation_command(LanguageActivationLevel.System, True)}
        """)
        print(message)


@click.command(name="language-container")
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucketfs-host', type=str, required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs-use-https', type=bool, default=False)
@click.option('--bucketfs-user', type=str, required=True, default="w")
@click.option('--bucketfs-password', prompt='bucketFS password', hide_input=True,
              default=lambda: os.environ.get(
                  utils.BUCKETFS_PASSWORD_ENVIRONMENT_VARIABLE, ""))
@click.option('--bucket', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True, default=None)
@click.option('--container-file',
              type=click.Path(exists=True, file_okay=True), default=None)
@click.option('--version', type=str, default=None)
@click.option('--dsn', type=str, required=True)
@click.option('--db-user', type=str, required=True)
@click.option('--db-pass', prompt='db password', hide_input=True,
              default=lambda: os.environ.get(
                  utils.DB_PASSWORD_ENVIRONMENT_VARIABLE, ""))
@click.option('--language-alias', type=str, default="PYTHON3_TE")
@click.option('--ssl-cert-path', type=str, default="")
@click.option('--use-ssl-cert-validation/--no-use-ssl-cert-validation', type=bool, default=True)
@click.option('--upload-container/--no-upload_container', type=bool, default=True)
@click.option('--alter-system/--no-alter-system', type=bool, default=True)
@click.option('--allow-override/--disallow-override', type=bool, default=False)
def language_container_deployer_main(
        bucketfs_name: str,
        bucketfs_host: str,
        bucketfs_port: int,
        bucketfs_use_https: bool,
        bucketfs_user: str,
        bucketfs_password: str,
        bucket: str,
        path_in_bucket: str,
        container_file: str,
        version: str,
        dsn: str,
        db_user: str,
        db_pass: str,
        language_alias: str,
        ssl_cert_path: str,
        use_ssl_cert_validation: bool,
        upload_container: bool,
        alter_system: bool,
        allow_override: bool):

    def call_runner():
        deployer = LanguageContainerDeployer.create(
            bucketfs_name=bucketfs_name,
            bucketfs_host=bucketfs_host,
            bucketfs_port=bucketfs_port,
            bucketfs_use_https=bucketfs_use_https,
            bucketfs_user=bucketfs_user,
            bucketfs_password=bucketfs_password,
            bucket=bucket,
            path_in_bucket=path_in_bucket,
            container_file=Path(container_file),
            dsn=dsn,
            db_user=db_user,
            db_password=db_pass,
            language_alias=language_alias,
            ssl_cert_path=ssl_cert_path,
            use_ssl_cert_validation=use_ssl_cert_validation)
        run_deployer(deployer, upload_container=upload_container, alter_system=alter_system,
                     allow_override=allow_override)

    if container_file:
        call_runner()
    elif version:
        with utils.get_container_file_from_github_release(version) as container:
            container_file = container
            call_runner()
    else:
        raise ValueError("You should specify either the release version to "
                         "download container file or the path of the already "
                         "downloaded container file.")


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    language_container_deployer_main()
