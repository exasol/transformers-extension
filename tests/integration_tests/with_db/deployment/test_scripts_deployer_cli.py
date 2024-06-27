from __future__ import annotations
from typing import Any

import pytest
from click.testing import CliRunner
from pyexasol import ExaConnection, ExaConnectionFailedError
import exasol.bucketfs as bfs

from tests.fixtures.language_container_fixture import LANGUAGE_ALIAS

from exasol_transformers_extension import deploy
from tests.utils.db_queries import DBQueries
from tests.utils.parameters import get_deploy_arg_list


def test_scripts_deployer_cli(backend,
                              deploy_params: dict[str, Any],
                              pyexasol_connection: ExaConnection,
                              request):
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    args_list = get_deploy_arg_list(deploy_params, schema_name, LANGUAGE_ALIAS)
    # We validate the server certificate in SaaS, but not in the Docker DB
    if backend == bfs.path.StorageBackend.saas:
        args_list.append("--use-ssl-cert-validation")
    else:
        args_list.append("--no-use-ssl-cert-validation")

    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 0
    assert DBQueries.check_all_scripts_deployed(
        pyexasol_connection, schema_name)


def test_scripts_deployer_cli_with_encryption_verify(backend,
                                                     deploy_params: dict[str, Any],
                                                     pyexasol_connection: ExaConnection,
                                                     request):
    if backend != bfs.path.StorageBackend.onprem:
        pytest.skip("We run this test only in the Docker-DB")
    schema_name = request.node.name
    pyexasol_connection.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

    args_list = get_deploy_arg_list(deploy_params, schema_name, LANGUAGE_ALIAS)
    args_list.append("--use-ssl-cert-validation")
    expected_exception_message = '[SSL: CERTIFICATE_VERIFY_FAILED]'
    runner = CliRunner()
    result = runner.invoke(deploy.main, args_list)
    assert result.exit_code == 1
    assert expected_exception_message in result.exception.args[0].message
    assert isinstance(result.exception, ExaConnectionFailedError)

