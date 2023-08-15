from collections import deque
from pathlib import Path, PosixPath
from urllib.parse import urlparse

import pytest
import transformers
from click.testing import CliRunner
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from pytest_itde.config import TestConfig

from exasol_transformers_extension import upload_model
from exasol_transformers_extension.utils import bucketfs_operations
from tests.utils import postprocessing
from tests.utils.parameters import bucketfs_params, model_params


@pytest.fixture(scope='function')
def download_sample_models(tmp_path) -> Path:
    for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
        downloader.from_pretrained(model_params.tiny_model, cache_dir=tmp_path)

    yield tmp_path


def adapt_file_to_upload(path: PosixPath, download_path: PosixPath):
    if path.is_dir():
        path = path / "not_empty"
    if ".no_exist" in path.parts:
        parts = list(path.parts)
        parts[path.parts.index(".no_exist")] = "no_exist"
        path = PosixPath(*parts)
    path = path.relative_to(download_path)
    return PosixPath(path)


def test_model_upload(download_sample_models: Path, bucketfs_location: BucketFSLocation, itde: TestConfig):
    sub_dir = 'sub_dir'
    download_path = download_sample_models
    upload_path = str(bucketfs_operations.get_model_path(
        sub_dir, model_params.tiny_model))
    parsed_url = urlparse(itde.bucketfs.url)
    host = parsed_url.netloc.split(":")[0]
    port = parsed_url.netloc.split(":")[1]
    args_list = [
        "--bucketfs-name", bucketfs_params.name,
        "--bucketfs-host", host,
        "--bucketfs-port", port,
        "--bucketfs_use-https", False,
        "--bucketfs-user", itde.bucketfs.username,
        "--bucketfs-password", itde.bucketfs.password,
        "--bucket", bucketfs_params.bucket,
        "--path-in-bucket", bucketfs_params.path_in_bucket,
        "--model-name", model_params.tiny_model,
        "--sub-dir", sub_dir,
        "--model-path", str(download_path),
        "--tokenizer-path", str(download_path)
    ]

    try:
        runner = CliRunner()
        result = runner.invoke(upload_model.main, args_list)
        assert result.exit_code == 0

        downloaded_files = set(adapt_file_to_upload(i, download_path) for i in download_path.rglob("*"))
        uploaded_files = set(PosixPath(i) for i in bucketfs_location.list_files_in_bucketfs(upload_path))
        assert uploaded_files == downloaded_files
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, upload_path)
