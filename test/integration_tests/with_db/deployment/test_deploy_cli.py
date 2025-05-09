from test.integration_tests.with_db.test_upload_model import run_model_upload_test

import exasol.bucketfs as bfs
import pyexasol
from click.testing import CliRunner
from exasol.python_extension_common.cli.std_options import (
    StdParams,
    get_cli_arg,
)
from exasol.python_extension_common.deployment.language_container_validator import (
    temp_schema,
)

from exasol_transformers_extension.deploy import (
    BUCKETFS_CONN_NAME_ARG,
    deploy_command,
)
from exasol_transformers_extension.deployment.language_container import export_slc

PATH_IN_BUCKET = "te_end2end"
BUCKETFS_CONN_NAME = "TE_E2E_BFS_CONN"
LANGUAGE_ALIAS = "TE_E2E_LANG_ALIAS"


def test_deploy_cli(
    backend,
    pyexasol_connection,
    backend_aware_database_params,
    backend_aware_bucketfs_params,
    bucketfs_cli_args,
    cli_args,
):
    """
    This test performs an installation of the extension.
    It then runs the model upload test to verify that the installation
    has brought the system into a ready-to-use state.
    """

    with temp_schema(pyexasol_connection) as schema_name:
        with export_slc() as container_file:
            args_string = " ".join(
                [
                    cli_args,
                    get_cli_arg(StdParams.schema, schema_name),
                    get_cli_arg(StdParams.container_file, container_file),
                    get_cli_arg(StdParams.language_alias, LANGUAGE_ALIAS),
                    get_cli_arg(StdParams.path_in_bucket, PATH_IN_BUCKET),
                    get_cli_arg(BUCKETFS_CONN_NAME_ARG, BUCKETFS_CONN_NAME),
                ]
            )
            runner = CliRunner()
            result = runner.invoke(
                deploy_command, args=args_string, catch_exceptions=False
            )
            assert result.exit_code == 0

            # This is a temporary workaround for the problem with slow slc file extraction
            # at a SaaS database. To be removed when a proper completion check is in place.
            if backend == "saas":
                import time

                time.sleep(30)

            db_conn = pyexasol.connect(
                **backend_aware_database_params, schema=schema_name
            )
            bfs_path = bfs.path.build_path(
                **backend_aware_bucketfs_params, path=PATH_IN_BUCKET
            )
            run_model_upload_test(
                bucketfs_cli_args, db_conn, bfs_path, BUCKETFS_CONN_NAME
            )
