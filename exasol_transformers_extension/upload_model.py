import os
import click
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.deployment import deployment_utils as utils


@click.command()
@click.option('--model-name', type=str, required=True,
              help="name of the model")
@click.option('--sub-dir', type=str, required=True,
              help="directory where the model is stored in the BucketFS")
@click.option('--model-path', type=click.Path(exists=True, file_okay=True),
              required=True, help="local path where model is located")
@click.option('--tokenizer-path', type=click.Path(exists=True, file_okay=True),
              required=True, help="local path where tokenizer model is located")
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucketfs-host', type=str, required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs_use-https', type=bool, default=False)
@click.option('--bucketfs-user', type=str, required=True, default="w")
@click.option('--bucketfs-password', prompt='bucketFS password', hide_input=True,
              default=lambda: os.environ.get(
                  utils.BUCKETFS_PASSWORD_ENVIRONMENT_VARIABLE, ""))
@click.option('--bucket', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True, default=None)
def main(
        bucketfs_name: str,
        bucketfs_host: str,
        bucketfs_port: int,
        bucketfs_use_https: bool,
        bucketfs_user: str,
        bucketfs_password: str,
        bucket: str,
        path_in_bucket: str,
        model_name: str,
        sub_dir: str,
        model_path: str,
        tokenizer_path: str):

    # create bucketfs location
    bucketfs_location = bucketfs_operations.create_bucketfs_location(
        bucketfs_name, bucketfs_host, bucketfs_port, bucketfs_use_https,
        bucketfs_user, bucketfs_password, bucket, path_in_bucket)

    # upload the downloaded model files into bucketfs
    upload_path = bucketfs_operations.get_model_path(sub_dir, model_name)
    for local_path in [model_path, tokenizer_path]:
        bucketfs_operations.upload_model_files_to_bucketfs(
            local_path, upload_path, bucketfs_location)


if __name__ == '__main__':
    main()
