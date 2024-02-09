from pathlib import Path, PosixPath
from urllib.parse import urlparse

import pytest
import transformers
from click.testing import CliRunner
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from pytest_itde import config

from exasol_transformers_extension import upload_model
from exasol_transformers_extension.utils import bucketfs_operations
from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils import postprocessing
from tests.utils.parameters import bucketfs_params, model_params
from tests.fixtures.model_fixture import download_model


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
                      bucketfs_location: BucketFSLocation, bucketfs_config: config.BucketFs):
    sub_dir = 'sub_dir'
    model_name = model_params.base_model
    download_path = download_model(model_name, tmp_path)
    upload_path = bucketfs_operations.get_model_path_with_pretrained(
        sub_dir, model_name)
    parsed_url = urlparse(bucketfs_config.url)
    host = parsed_url.netloc.split(":")[0]
    port = parsed_url.netloc.split(":")[1]
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
        "--local-model-path", str(download_path),
    ]

    try:
        runner = CliRunner()
        result = runner.invoke(upload_model.main, args_list)
        assert result.exit_code == 0
        assert str(upload_path.with_suffix(".tar.gz")) in bucketfs_location.list_files_in_bucketfs(".")

        bucketfs_conn_name, schema_name = setup_database
        text_data = "Exasol is an analytics <mask> management software company."
        input_data = [
            (
                '',
                bucketfs_conn_name,
                None,
                sub_dir,
                model_name,
                text_data,
                1
            )
        ]

        query = f"SELECT TE_FILLING_MASK_UDF(" \
                f"t.device_id, " \
                f"t.bucketfs_conn_name, " \
                f"t.token_conn_name, " \
                f"t.sub_dir, " \
                f"t.model_name, " \
                f"t.text_data," \
                f"t.top_k" \
                f") FROM (VALUES {python_rows_to_sql(input_data)} " \
                f"AS t(device_id, bucketfs_conn_name, token_conn_name, sub_dir, " \
                f"model_name, text_data, top_k));"

        # execute sequence classification UDF
        result = pyexasol_connection.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, sub_dir)
