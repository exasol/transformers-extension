import click
from pathlib import Path
from exasol_transformers_extension.deployment import utils
from exasol_transformers_extension.deployment.language_container_deployer import \
    LanguageContainerDeployer


@click.command(name="language-container")
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucketfs-host', type=str, required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs_use-https', type=bool, default=False)
@click.option('--bucketfs-user', type=str, required=True, default="w")
@click.option('--bucketfs-password', type=str)
@click.option('--bucket', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True, default=None)
@click.option('--container-file',
              type=click.Path(exists=True, file_okay=True), default=None)
@click.option('--version', type=str, default=None)
@click.option('--dsn', type=str, required=True)
@click.option('--db-user', type=str, required=True)
@click.option('--db-pass', type=str)
@click.option('--language-alias', type=str, default="PYTHON3_TE")
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
        language_alias: str):

    def call_runner():
        LanguageContainerDeployer.run(
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
            db_password=db_password,
            language_alias=language_alias
        )

    bucketfs_password = utils.get_password(
        bucketfs_password, bucketfs_user,
        utils.BUCKETFS_PASSWORD_ENVIRONMENT_VARIABLE, "BucketFS Password")
    db_password = utils.get_password(
        db_pass, db_user,
        utils.DB_PASSWORD_ENVIRONMENT_VARIABLE, "DB Password")

    if container_file:
        call_runner()
    elif version:
        with utils.get_container_file_from_github_release(version) as container:
            container_file = container
            call_runner()
    else:
        raise Exception("You should specify either the release version to "
                        "download container file or the path of the already "
                        "downloaded container file.")


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    language_container_deployer_main()
