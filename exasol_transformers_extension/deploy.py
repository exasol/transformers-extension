"""Deploy command which installs Transformers extension SLC,
scripts, creates connection objects"""

import logging

import click
from exasol.python_extension_common.cli.bucketfs_conn_object_cli import (
    BucketfsConnObjectCli,
)
from exasol.python_extension_common.cli.language_container_deployer_cli import (
    LanguageContainerDeployerCli,
)
from exasol.python_extension_common.cli.std_options import (
    ParameterFormatters,
    StdParams,
    StdTags,
    make_option_secret,
    select_std_options,
)
from exasol.python_extension_common.connections.bucketfs_location import (
    ConnectionInfo,
    write_bucketfs_conn_object,
)
from exasol.python_extension_common.connections.pyexasol_connection import (
    open_pyexasol_connection,
)

from exasol_transformers_extension.deployment.scripts_deployer import ScriptsDeployer
from exasol_transformers_extension.deployment.te_language_container_deployer import (
    TeLanguageContainerDeployer,
)

DEPLOY_SLC_ARG = "deploy_slc"
DEPLOY_SCRIPTS_ARG = "deploy_scripts"
CONTAINER_URL_ARG = "container_url"
CONTAINER_NAME_ARG = "container_name"
BUCKETFS_CONN_NAME_ARG = "bucketfs_conn_name"
TOKEN_CONN_NAME_ARG = "token_conn_name"
TOKEN_ARG = "token"

def version_formatters() -> ParameterFormatters:
    formatters = ParameterFormatters()
    formatters.set_formatter(
        CONTAINER_URL_ARG, TeLanguageContainerDeployer.SLC_URL_FORMATTER
    )
    formatters.set_formatter(CONTAINER_NAME_ARG, TeLanguageContainerDeployer.SLC_NAME)
    return formatters


# If the version is specified, then TE will infer the container name and
# download URL.  Otherwise the user needs to provide dedicated CLI options.
# Variable `formatters` is used to propagate the value of StdParams.version to
# dependent CLI parameters.
formatters = {StdParams.version: version_formatters()}


def get_opt_name(arg_name: str) -> str:
    """get opt_name for arg_name"""
    # This and the next function should have been implemented in the PEC.
    # #todo make ticket and remove this comment
    return f'--{arg_name.replace("_", "-")}'


def get_bool_opt_name(arg_name: str) -> str:
    """turn arg_name into bool_opt_name"""
    opt_name = arg_name.replace("_", "-")
    return f"--{opt_name}/--no-{opt_name}"


opt_lang_alias = {"type": str, "default": "PYTHON3_TE"}
opt_token = {"type": str, "help": "Huggingface token for private models"}
make_option_secret(opt_token, prompt="Huggingface token")
opts = select_std_options(
    [StdTags.DB, StdTags.BFS, StdTags.SLC],
    formatters=formatters,
    override={StdParams.language_alias: opt_lang_alias},
)
opts.append(
    click.Option(
        [get_bool_opt_name(DEPLOY_SLC_ARG)], type=bool, default=True, help="Deploy SLC"
    )
)
opts.append(
    click.Option(
        [get_bool_opt_name(DEPLOY_SCRIPTS_ARG)],
        type=bool,
        default=True,
        help="Deploy scripts",
    )
)
opts.append(
    click.Option(
        [get_opt_name(BUCKETFS_CONN_NAME_ARG)],
        type=str,
        help="Create BucketFS connection object with this name",
    )
)
opts.append(
    click.Option(
        [get_opt_name(TOKEN_CONN_NAME_ARG)],
        type=str,
        help="Create token connection object with this name",
    )
)

opts.append(click.Option([get_opt_name(TOKEN_ARG)], **opt_token))  # type: ignore


def deploy(**kwargs):
    """Deploy TE slc, scripts, create bucketfs connection object and
    token connection object."""
    # Deploy the SLC
    if kwargs[DEPLOY_SLC_ARG]:
        slc_deployer = LanguageContainerDeployerCli(
            container_url_arg=CONTAINER_URL_ARG, container_name_arg=CONTAINER_NAME_ARG
        )

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
        conn_info = ConnectionInfo(address="", user="", password=kwargs[TOKEN_ARG])
        pyexasol_conn = open_pyexasol_connection(**kwargs)
        # Very badly named function. 'bucketfs' should not be in the name.
        write_bucketfs_conn_object(
            pyexasol_conn, kwargs[TOKEN_CONN_NAME_ARG], conn_info
        )


deploy_command = click.Command(None, params=opts, callback=deploy)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s  - %(message)s", level=logging.DEBUG
    )

    deploy_command()
