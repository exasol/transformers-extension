from pathlib import Path, PosixPath
from urllib.parse import urlparse

import pytest
import transformers
from click.testing import CliRunner
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from pytest_itde import config

from exasol_transformers_extension import upload_model
from exasol_transformers_extension.utils import bucketfs_operations
from tests.utils import postprocessing
from tests.utils.parameters import bucketfs_params, model_params


@pytest.fixture(scope='function')
def download_sample_models(tmp_path) -> Path:
    for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
        downloader.from_pretrained(model_params.base_model, cache_dir=tmp_path)

    yield tmp_path, model_params.base_model


def adapt_file_to_upload(path: PosixPath, download_path: PosixPath):
    if path.is_dir():
        path = path / "not_empty"
    if ".no_exist" in path.parts:
        parts = list(path.parts)
        parts[path.parts.index(".no_exist")] = "no_exist"
        path = PosixPath(*parts)
    path = path.relative_to(download_path)
    return PosixPath(path)


def test_model_upload(setup_database, pyexasol_connection, download_sample_models: Path,
                      bucketfs_location: BucketFSLocation, bucketfs_config: config.BucketFs):
    sub_dir = 'sub_dir'
    download_path, model_name = download_sample_models
    upload_path = bucketfs_operations.get_model_path(
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
        input_data = (
            '',
            bucketfs_conn_name,
            sub_dir,
            model_name,
            text_data,
            1)

        query = f"SELECT TE_FILLING_MASK_UDF(" \
                f"t.device_id, " \
                f"t.bucketfs_conn_name, " \
                f"t.sub_dir, " \
                f"t.model_name, " \
                f"t.text_data," \
                f"t.top_k" \
                f") FROM (VALUES {str(input_data)} " \
                f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
                f"model_name, text_data, top_k));"

        # execute sequence classification UDF
        result = pyexasol_connection.execute(query).fetchall()
        assert len(result) == 1 and result[0][-1] is None
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, sub_dir)
