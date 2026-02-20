import time
from pathlib import (
    Path,
    PosixPath,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils import postprocessing
from test.utils.parameters import model_params

import exasol.bucketfs as bfs
import pyexasol
from click.testing import CliRunner
from exasol.python_extension_common.cli.std_options import (
    StdParams,
    get_cli_arg,
)

from exasol_transformers_extension.upload_model import (
    MODEL_NAME_ARG,
    SUBDIR_ARG,
    TASK_TYPE_ARG,
    upload_model_command,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
)


def adapt_file_to_upload(path: PosixPath, download_path: PosixPath):
    if path.is_dir():
        path = path / "not_empty"
    if ".no_exist" in path.parts:
        parts = list(path.parts)
        parts[path.parts.index(".no_exist")] = "no_exist"
        path = PosixPath(*parts)
    path = path.relative_to(download_path)
    return PosixPath(path)


def run_model_upload_test(
    bucketfs_cli_args,
    db_conn: pyexasol.ExaConnection,
    bucketfs_location: bfs.path.PathLike,
    bucketfs_conn_name: str,
):
    sub_dir = "sub_dir"
    model_specification = model_params.fill_model_specs
    model_name = model_specification.model_name
    current_model_specs = get_BucketFSModelSpecification_from_model_Specs(
        model_specification, "", Path(sub_dir)
    )
    upload_path = current_model_specs.get_bucketfs_model_save_path()

    args_string = " ".join(
        [
            bucketfs_cli_args,
            get_cli_arg(StdParams.path_in_bucket, str(bucketfs_location)),
            get_cli_arg(MODEL_NAME_ARG, model_name),
            get_cli_arg(SUBDIR_ARG, sub_dir),
            get_cli_arg(TASK_TYPE_ARG, "fill-mask"),
        ]
    )

    try:
        runner = CliRunner()
        result = runner.invoke(
            upload_model_command, args=args_string, catch_exceptions=False
        )
        if result.exit_code != 0:
            print("Exception:", result.exception)
            print("ExcInfo:", result.exc_info)
            print("STDERR:", result.stderr_bytes)
            print("STDOUT:", result.stdout_bytes)
        assert result.exit_code == 0
        time.sleep(20)
        bucketfs_upload_location = bucketfs_location / upload_path.with_suffix(
            ".tar.gz"
        )
        assert bucketfs_upload_location.is_file()

        text_data = "Exasol is an analytics <mask> management software company."
        input_data = [("", bucketfs_conn_name, sub_dir, model_name, text_data, 1)]

        query = (
            f"SELECT AI_FILL_MASK_EXTENDED("
            f"t.device_id, "
            f"t.bucketfs_conn_name, "
            f"t.sub_dir, "
            f"t.model_name, "
            f"t.text_data,"
            f"t.top_k"
            f") FROM (VALUES {python_rows_to_sql(input_data)} "
            f"AS t(device_id, bucketfs_conn_name, sub_dir, "
            f"model_name, text_data, top_k));"
        )

        # execute fill mask UDF
        result = db_conn.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, sub_dir)


def test_model_upload(bucketfs_cli_args, setup_database, db_conn, bucketfs_location):
    bucketfs_conn_name, _ = setup_database
    run_model_upload_test(
        bucketfs_cli_args, db_conn, bucketfs_location, bucketfs_conn_name
    )
