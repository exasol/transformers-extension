from pathlib import Path, PosixPath
from urllib.parse import urlparse

from click.testing import CliRunner
from pytest_itde import config
import exasol.bucketfs as bfs

from exasol_transformers_extension import upload_model as upload_model_cli
from exasol_transformers_extension.utils.bucketfs_model_specification import get_BucketFSModelSpecification_from_model_Specs
from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils import postprocessing
from tests.utils.parameters import bucketfs_params, model_params

from tests.fixtures.model_fixture import *
from tests.fixtures.setup_database_fixture import *
from tests.fixtures.language_container_fixture import *
from tests.fixtures.bucketfs_fixture import *
from tests.fixtures.database_connection_fixture import *

def adapt_file_to_upload(path: PosixPath, download_path: PosixPath):
    if path.is_dir():
        path = path / "not_empty"
    if ".no_exist" in path.parts:
        parts = list(path.parts)
        parts[path.parts.index(".no_exist")] = "no_exist"
        path = PosixPath(*parts)
    path = path.relative_to(download_path)
    return PosixPath(path)


def test_model_upload(setup_database, pyexasol_connection, tmp_path: Path,
                      bucketfs_location: bfs.path.PathLike, bucketfs_config: config.BucketFs):
    sub_dir = 'sub_dir'
    model_specification = model_params.base_model_specs
    model_specification.task_type = "filling_mask"
    model_name = model_specification.model_name
    current_model_specs = get_BucketFSModelSpecification_from_model_Specs(model_specification, "", Path(sub_dir))
    upload_path = current_model_specs.get_bucketfs_model_save_path()
    parsed_url = urlparse(bucketfs_config.url)
    host = parsed_url.netloc.split(":")[0]
    port = parsed_url.netloc.split(":")[1]
    print("path in bucket: " + bucketfs_params.path_in_bucket)
    args_list = [
        "--bucketfs-name", bucketfs_params.name,
        "--bucketfs-host", host,
        "--bucketfs-port", port,
        "--bucketfs-use-https", False,
        "--bucketfs-user", bucketfs_config.username,
        "--bucketfs-password", bucketfs_config.password,
        "--bucket", bucketfs_params.bucket,
        "--path-in-bucket", bucketfs_params.path_in_bucket,
        "--model-name", model_name,
        "--sub-dir", sub_dir,
        "--task_type", "filling_mask"
    ]

    try:
        runner = CliRunner()
        result = runner.invoke(upload_model_cli.main, args_list)
        print(result)
        assert result.exit_code == 0
        bucketfs_upload_location = bucketfs_location / upload_path.with_suffix(".tar.gz")
        assert bucketfs_upload_location.is_file()

        bucketfs_conn_name, schema_name = setup_database
        text_data = "Exasol is an analytics <mask> management software company."
        input_data = [
            (
                '',
                bucketfs_conn_name,
                sub_dir,
                model_name,
                text_data,
                1
            )
        ]

        query = f"SELECT TE_FILLING_MASK_UDF(" \
                f"t.device_id, " \
                f"t.bucketfs_conn_name, " \
                f"t.sub_dir, " \
                f"t.model_name, " \
                f"t.text_data," \
                f"t.top_k" \
                f") FROM (VALUES {python_rows_to_sql(input_data)} " \
                f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
                f"model_name, text_data, top_k));"

        # execute sequence classification UDF
        result = pyexasol_connection.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, sub_dir)