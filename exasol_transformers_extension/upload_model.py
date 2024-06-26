from __future__ import annotations
from pathlib import Path

import click

from exasol.python_extension_common.deployment.language_container_deployer_cli import (
    SECRET_DISPLAY, SecretParams, secret_callback)
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.deployment import deployment_utils as utils
from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification


@click.command()
@click.option('--model-name', type=str, required=True,
              help="name of the model")
@click.option('--task_type', type=str, required=True) #todo change docu (needed to know where to safe model)
@click.option('--sub-dir', type=str, required=True,
              help="directory where the model is stored in the BucketFS")
@click.option('--local-model-path', type=click.Path(exists=True, file_okay=True),
              required=True, help="local path where model is located")
@click.option('--bucketfs-name', type=str)
@click.option('--bucketfs-host', type=str)
@click.option('--bucketfs-port', type=int)
@click.option('--bucketfs-use-https', type=bool, default=False)
@click.option('--bucketfs-user', type=str, default="w")
@click.option(f'--{SecretParams.BUCKETFS_PASSWORD.value}', type=str,
              prompt='BucketFS password', prompt_required=False,
              hide_input=True, default=SECRET_DISPLAY, callback=secret_callback)
@click.option('--bucket', type=str)
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
@click.option('--path-in-bucket', type=str, required=True, default=None)
@click.option('--use-ssl-cert-validation/--no-use-ssl-cert-validation', type=bool, default=True)
def main(
        model_name: str,
        task_type: str,
        sub_dir: str,
        local_model_path: str,
        bucketfs_name: str,
        bucketfs_host: str,
        bucketfs_port: int,
        bucketfs_use_https: bool,
        bucketfs_user: str,
        bucketfs_password: str | None,
        bucket: str | None,
        saas_url: str | None,
        saas_account_id: str | None,
        saas_database_id: str | None,
        saas_database_name: str | None,
        saas_token: str | None,
        path_in_bucket: str,
        use_ssl_cert_validation: bool) -> None:
    """
    Script for uploading locally saved model files to BucketFS. Files should have been saved locally
    using Transformers save_pretrained function. This ensures proper loading from the BucketFS later
    """
    # create bucketfs location
    bucketfs_location = bucketfs_operations.create_bucketfs_location(
        bucketfs_name=bucketfs_name,
        bucketfs_host=bucketfs_host,
        bucketfs_port=bucketfs_port,
        bucketfs_use_https=bucketfs_use_https,
        bucketfs_user=bucketfs_user,
        bucketfs_password=bucketfs_password,
        bucket=bucket,
        saas_url=saas_url,
        saas_account_id=saas_account_id,
        saas_database_id=saas_database_id,
        saas_database_name=saas_database_name,
        saas_token=saas_token,
        path_in_bucket=path_in_bucket,
        use_ssl_cert_validation=use_ssl_cert_validation)

    # create CurrentModelSpecification for model to be loaded
    current_model_specs = CurrentModelSpecification(model_name, task_type, "", Path(sub_dir))
    # upload the downloaded model files into bucketfs
    upload_path = current_model_specs.get_bucketfs_model_save_path()
    bucketfs_operations.upload_model_files_to_bucketfs(
        local_model_path, upload_path, bucketfs_location)


if __name__ == '__main__':
    main()
