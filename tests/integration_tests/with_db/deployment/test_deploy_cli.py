import pytest
from click.testing import CliRunner
from pyexasol import ExaConnectionFailedError
from exasol.pytest_backend import BACKEND_ONPREM
from exasol.python_extension_common.cli.std_options import StdParams, get_cli_arg
from exasol.python_extension_common.deployment.language_container_validator import temp_schema

from exasol_transformers_extension.deploy import (
    deploy_command, DEPLOY_SLC_ARG, BUCKETFS_CONN_NAME_ARG, get_bool_opt_name)
from exasol_transformers_extension.deployment.language_container import export_slc


def test_deploy_cli(pyexasol_connection,
                    language_alias,
                    cli_args):

    with temp_schema(pyexasol_connection) as schema_name:
        with export_slc() as container_file:
            args_string = ' '.join([cli_args,
                                    get_cli_arg(StdParams.schema, schema_name),
                                    get_cli_arg(StdParams.container_file, container_file),
                                    get_cli_arg(StdParams.language_alias, language_alias),
                                    get_cli_arg(BUCKETFS_CONN_NAME_ARG, 'MY_BFS_CONN')])
            runner = CliRunner()
            result = runner.invoke(deploy_command, args=args_string, catch_exceptions=False)
            assert result.exit_code == 0


def test_scripts_deployer_cli_with_encryption_verify(backend,
                                                     pyexasol_connection,
                                                     language_alias,
                                                     cli_args,
                                                     deployed_slc):
    if backend != BACKEND_ONPREM:
        pytest.skip(("We run this test only with the Docker-DB "
                     "because SaaS always verifies the SSL certificate"))

    with temp_schema(pyexasol_connection) as schema_name:
        cert_validation_opt_name = get_bool_opt_name(StdParams.use_ssl_cert_validation.name)
        cert_validation_choice = cert_validation_opt_name.split('/')
        args_string = ' '.join([cli_args.replace(cert_validation_choice[1], cert_validation_choice[0]),
                                get_cli_arg(StdParams.schema, schema_name),
                                get_cli_arg(DEPLOY_SLC_ARG, False),
                                get_cli_arg(StdParams.language_alias, language_alias)])
        runner = CliRunner()
        result = runner.invoke(deploy_command, args=args_string, catch_exceptions=True)
        expected_exception_message = '[SSL: CERTIFICATE_VERIFY_FAILED]'
        assert result.exit_code == 1
        assert expected_exception_message in result.exception.args[0].message
        assert isinstance(result.exception, ExaConnectionFailedError)
