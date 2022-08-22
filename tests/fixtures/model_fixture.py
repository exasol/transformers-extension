from pathlib import PurePosixPath, Path

import pytest
import tempfile
import transformers
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_transformers_extension.utils import bucketfs_operations
from tests.utils.parameters import model_params


@pytest.fixture(scope="session")
def download_base_model() -> str:
    with tempfile.TemporaryDirectory() as tmpdir_name:
        for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
            downloader.from_pretrained(
                model_params.base, cache_dir=tmpdir_name)
        yield tmpdir_name


@pytest.fixture(scope="session")
def download_seq2seq_model() -> str:
    with tempfile.TemporaryDirectory() as tmpdir_name:
        for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
            downloader.from_pretrained(
                model_params.seq2seq, cache_dir=tmpdir_name)
        yield tmpdir_name


@pytest.fixture(scope="session")
def upload_model_to_local_bucketfs(
        download_base_model, download_seq2seq_model, request) -> PurePosixPath:

    model_name = request.param
    download_tmpdir_name = download_base_model \
        if model_name == model_params.base else download_seq2seq_model

    with tempfile.TemporaryDirectory() as upload_tmpdir_name:
        bucketfs_location = LocalFSMockBucketFSLocation(upload_tmpdir_name)

        model_path = bucketfs_operations.get_model_path(
            model_params.sub_dir, model_name)
        bucketfs_operations.upload_model_files_to_bucketfs(
            tmpdir_name=download_tmpdir_name,
            model_path=Path(model_path),
            bucketfs_location=bucketfs_location)

        yield upload_tmpdir_name


@pytest.fixture(scope="session")
def get_local_bucketfs_path() -> str:
    bucket_base_path = ''
    with tempfile.TemporaryDirectory() as tmpdir_name:
        model_path = PurePosixPath(tmpdir_name, bucket_base_path)

        yield str(model_path)


@pytest.fixture(scope="session")
def upload_model_to_bucketfs(
        download_base_model, download_seq2seq_model,
        request, bucketfs_location) -> Path:

    model_name = request.param
    tmpdir_name = download_base_model \
        if model_name == model_params.base else download_seq2seq_model

    model_path = bucketfs_operations.get_model_path(
        model_params.sub_dir, model_name)
    bucketfs_operations.upload_model_files_to_bucketfs(
        tmpdir_name, model_path, bucketfs_location)

    yield model_path

    bucketfs_files = bucketfs_location.list_files_in_bucketfs(str(model_path))
    for file_ in bucketfs_files:
        try:
            bucketfs_location.delete_file_in_bucketfs(
                str(PurePosixPath(model_path, file_)))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")
