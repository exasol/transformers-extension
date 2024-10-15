import logging
import click
from exasol.python_extension_common.cli.std_options import (
    StdParams, StdTags, select_std_options, ParameterFormatters)
from exasol.python_extension_common.cli.language_container_deployer_cli import (
    LanguageContainerDeployerCli)
from exasol.python_extension_common.cli.bucketfs_conn_object_cli import BucketfsConnObjectCli
from exasol.python_extension_common.connections.pyexasol_connection import open_pyexasol_connection
from exasol.python_extension_common.connections.bucketfs_location import (
    ConnectionInfo, write_bucketfs_conn_object)
from exasol_transformers_extension.deployment.te_language_container_deployer import (
    TeLanguageContainerDeployer)
from exasol_transformers_extension.deployment.scripts_deployer import ScriptsDeployer

DEPLOY_SLC_ARG = 'deploy_slc'
DEPLOY_SCRIPTS_ARG = 'deploy_scripts'
CONTAINER_URL_ARG = 'container_url'
CONTAINER_NAME_ARG = 'container_name'
BUCKETFS_CONN_NAME_ARG = 'bucketfs_conn_name'
TOKEN_CONN_NAME_ARG = 'token_conn_name'
TOKEN_ARG = 'model-token'

ver_formatter = ParameterFormatters()
ver_formatter.set_formatter(CONTAINER_URL_ARG, TeLanguageContainerDeployer.SLC_URL_FORMATTER)
ver_formatter.set_formatter(CONTAINER_NAME_ARG, TeLanguageContainerDeployer.SLC_NAME)
formatters = {StdParams.version: ver_formatter}


def get_opt_name(arg_name: str) -> str:
    # This and the next function should have been implemented in the PEC.
    return f'--{arg_name.replace("_", "-")}'


def get_bool_opt_name(arg_name: str) -> str:
    opt_name = arg_name.replace("_", "-")
    return f'--{opt_name}/--no-{opt_name}'


opts = select_std_options([StdTags.DB, StdTags.BFS, StdTags.SLC], formatters=formatters)
opts.append(click.Option([get_bool_opt_name(DEPLOY_SLC_ARG)], type=bool, default=True))
opts.append(click.Option([get_bool_opt_name(DEPLOY_SCRIPTS_ARG)], type=bool, default=True))
opts.append(click.Option([get_opt_name(BUCKETFS_CONN_NAME_ARG)], type=str))
opts.append(click.Option([get_opt_name(TOKEN_CONN_NAME_ARG)], type=str))
opts.append(click.Option([get_opt_name(TOKEN_ARG)], type=str))


def deploy(**kwargs):

    # Make sure there is a valid language_alias
    if not kwargs.get(StdParams.language_alias.name):
        kwargs[StdParams.language_alias.name] = 'PYTHON3_TE'

    # Deploy the SLC
    if kwargs[DEPLOY_SLC_ARG]:
        slc_deployer = LanguageContainerDeployerCli(
            container_url_arg=CONTAINER_URL_ARG,
            container_name_arg=CONTAINER_NAME_ARG)

        slc_deployer(**kwargs)

    # Deploy the scripts
    if kwargs[DEPLOY_SCRIPTS_ARG]:
        ScriptsDeployer.run(**kwargs)

    # Create bucketfs connection object
    if kwargs[BUCKETFS_CONN_NAME_ARG]:
        bucketfs_conn_deployer = BucketfsConnObjectCli(BUCKETFS_CONN_NAME_ARG)
        bucketfs_conn_deployer(**kwargs)

    # Create token connection object
    if kwargs[TOKEN_CONN_NAME_ARG] and kwargs[TOKEN_ARG]:
        conn_info = ConnectionInfo(address='', user='', password=kwargs[TOKEN_ARG])
        pyexasol_conn = open_pyexasol_connection(**kwargs)
        # Very badly named function. 'bucketfs' should not be in the name.
        write_bucketfs_conn_object(pyexasol_conn, kwargs[TOKEN_CONN_NAME_ARG], conn_info)


deploy_command = click.Command(None, params=opts, callback=deploy)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(module)s  - %(message)s',
        level=logging.DEBUG)

    deploy_command()
