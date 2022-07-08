from pathlib import PurePosixPath
import pytest
import tempfile
import transformers
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_transformers_extension.udfs import bucketfs_operations
from tests.utils.parameters import model_params


@pytest.fixture(scope="session")
def download_model() -> str:
    with tempfile.TemporaryDirectory() as tmpdir_name:
        for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
            downloader.from_pretrained(model_params.name, cache_dir=tmpdir_name)
        yield tmpdir_name


@pytest.fixture(scope="session")
def upload_model_to_local_bucketfs(download_model):
    with tempfile.TemporaryDirectory() as upload_tmpdir_name:
        model_path = PurePosixPath(
            upload_tmpdir_name,
            bucketfs_operations.get_model_path(model_params.name))
        bucketfs_location = LocalFSMockBucketFSLocation(model_path)

        downloaded_tmpdir_name = download_model
        bucketfs_operations.upload_model_files_to_bucketfs(
            tmpdir_name=downloaded_tmpdir_name,
            model_path=str(model_path),
            bucketfs_location=bucketfs_location)

        yield model_path


@pytest.fixture(scope="session")
def upload_dummy_model_to_local_bucketfs():
    model_file_data_map = {
        "model_file1.txt": "Sample data in model_file1.txt",
        "model_file2.txt": "Sample data in model_file1.txt"}

    with tempfile.TemporaryDirectory() as tmpdir_name:
        model_path = PurePosixPath(
            tmpdir_name, bucketfs_operations.get_model_path(model_params.name))
        bucketfs_location = LocalFSMockBucketFSLocation(model_path)

        for file_name, content in model_file_data_map.items():
            bucketfs_location.upload_string_to_bucketfs(
                str(PurePosixPath(tmpdir_name, file_name)), content)

        yield model_path


@pytest.fixture(scope="session")
def upload_model_to_bucketfs(download_model, bucketfs_location):
    tmpdir_name = download_model
    model_path = bucketfs_operations.get_model_path(model_params.name)

    bucketfs_operations.upload_model_files_to_bucketfs(
        tmpdir_name, model_path, bucketfs_location)

    yield

    bucketfs_files = bucketfs_location.list_files_in_bucketfs(model_path)
    for file_ in bucketfs_files:
        try:
            bucketfs_location.delete_file_in_bucketfs(
                str(PurePosixPath(model_path, file_)))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")
