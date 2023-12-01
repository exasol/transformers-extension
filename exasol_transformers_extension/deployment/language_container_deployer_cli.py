import os
import click
from pathlib import Path
from exasol_transformers_extension.deployment import deployment_utils as utils
from exasol_transformers_extension.deployment.te_language_container_deployer import TeLanguageContainerDeployer


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
@click.option('--ssl-client-cert-path', type=str, default="")
@click.option('--ssl-client-private-key', type=str, default="")
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
        ssl_client_cert_path: str,
        ssl_client_private_key: str,
        use_ssl_cert_validation: bool,
        upload_container: bool,
        alter_system: bool,
        allow_override: bool):

    deployer = TeLanguageContainerDeployer.create(
        bucketfs_name=bucketfs_name,
        bucketfs_host=bucketfs_host,
        bucketfs_port=bucketfs_port,
        bucketfs_use_https=bucketfs_use_https,
        bucketfs_user=bucketfs_user,
        bucketfs_password=bucketfs_password,
        bucket=bucket,
        path_in_bucket=path_in_bucket,
        dsn=dsn,
        db_user=db_user,
        db_password=db_pass,
        language_alias=language_alias,
        ssl_trusted_ca=ssl_cert_path,
        ssl_client_certificate=ssl_client_cert_path,
        ssl_private_key=ssl_client_private_key,
        use_ssl_cert_validation=use_ssl_cert_validation)

    if not upload_container:
        deployer.run(alter_system=alter_system, allow_override=allow_override)
    elif container_file:
        deployer.run(container_file=Path(container_file), alter_system=alter_system, allow_override=allow_override)
    elif version:
        deployer.download_from_git_and_run(version, alter_system=alter_system, allow_override=allow_override)
    else:
        raise ValueError("To upload a language container you should specify either its "
                         "release version or a path of the already downloaded container file.")


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    language_container_deployer_main()
