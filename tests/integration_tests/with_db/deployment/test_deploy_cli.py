import pytest
from click.testing import CliRunner
import pyexasol
import exasol.bucketfs as bfs
from exasol.pytest_backend import BACKEND_ONPREM
from exasol.python_extension_common.cli.std_options import StdParams, get_cli_arg
from exasol.python_extension_common.deployment.language_container_validator import temp_schema

from exasol_transformers_extension.deploy import (
    deploy_command, DEPLOY_SLC_ARG, BUCKETFS_CONN_NAME_ARG, get_bool_opt_name)
from exasol_transformers_extension.deployment.language_container import export_slc
from tests.integration_tests.with_db.deployment.test_upload_model import run_model_upload_test

PATH_IN_BUCKET = 'te_end2end'
BUCKETFS_CONN_NAME = 'TE_E2E_BFS_CONN'
LANGUAGE_ALIAS = 'TE_E2E_LANG_ALIAS'


@pytest.mark.skip('Need to sort out the model upload test first')
def test_deploy_cli(pyexasol_connection,
                    backend_aware_database_params,
                    backend_aware_bucketfs_params,
                    bucketfs_cli_args,
                    cli_args):

    with temp_schema(pyexasol_connection) as schema_name:
        with export_slc() as container_file:
            args_string = ' '.join([cli_args,
                                    get_cli_arg(StdParams.schema, schema_name),
                                    get_cli_arg(StdParams.container_file, container_file),
                                    get_cli_arg(StdParams.language_alias, LANGUAGE_ALIAS),
                                    get_cli_arg(StdParams.path_in_bucket, PATH_IN_BUCKET),
                                    get_cli_arg(BUCKETFS_CONN_NAME_ARG, BUCKETFS_CONN_NAME)])
            runner = CliRunner()
            result = runner.invoke(deploy_command, args=args_string, catch_exceptions=False)
            assert result.exit_code == 0

            db_conn = pyexasol.connect(**backend_aware_database_params, schema=schema_name)
            bfs_path = bfs.path.build_path(**backend_aware_bucketfs_params, path=PATH_IN_BUCKET)
            run_model_upload_test(bucketfs_cli_args, db_conn, bfs_path, BUCKETFS_CONN_NAME)


@pytest.mark.skip('Need to sort out the model upload test first')
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
        assert isinstance(result.exception, pyexasol.ExaConnectionFailedError)
