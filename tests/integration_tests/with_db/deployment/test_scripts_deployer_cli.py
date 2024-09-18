from __future__ import annotations
from typing import Any

import pytest
from click.testing import CliRunner
from pyexasol import ExaConnection, ExaConnectionFailedError
from exasol.pytest_backend import BACKEND_ONPREM, BACKEND_SAAS
from exasol.python_extension_common.deployment.language_container_validator import temp_schema

from tests.fixtures.language_container_fixture_constants import LANGUAGE_ALIAS

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries
from tests.utils.parameters import get_arg_list


def test_scripts_deployer_cli(backend,
                              deploy_params: dict[str, Any],
                              pyexasol_connection: ExaConnection,
                              upload_slc):

    with temp_schema(pyexasol_connection) as schema_name:
        args_list = get_arg_list(**deploy_params, schema=schema_name, language_alias=LANGUAGE_ALIAS)
        args_list.insert(0, "scripts")
        # We validate the server certificate in SaaS, but not in the Docker DB
        if backend == BACKEND_SAAS:
            args_list.append("--use-ssl-cert-validation")
        else:
            args_list.append("--no-use-ssl-cert-validation")

        runner = CliRunner()
        result = runner.invoke(deploy.main, args_list)
        assert not result.exception
        assert result.exit_code == 0
        assert DBQueries.check_all_scripts_deployed(
            pyexasol_connection, schema_name)


def test_scripts_deployer_cli_with_encryption_verify(backend,
                                                     deploy_params: dict[str, Any],
                                                     pyexasol_connection: ExaConnection,
                                                     upload_slc):
    if backend != BACKEND_ONPREM:
        pytest.skip(("We run this test only with the Docker-DB "
                     "because SaaS always verifies the SSL certificate"))

    with temp_schema(pyexasol_connection) as schema_name:
        args_list = get_arg_list(**deploy_params, schema=schema_name, language_alias=LANGUAGE_ALIAS)
        args_list.insert(0, "scripts")
        args_list.append("--use-ssl-cert-validation")
        expected_exception_message = '[SSL: CERTIFICATE_VERIFY_FAILED]'
        runner = CliRunner()
        result = runner.invoke(deploy.main, args_list)
        assert result.exit_code == 1
        assert expected_exception_message in result.exception.args[0].message
        assert isinstance(result.exception, ExaConnectionFailedError)
