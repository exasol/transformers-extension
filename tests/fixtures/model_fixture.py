import pytest
import tempfile
import transformers
from contextlib import contextmanager
from pathlib import PurePosixPath, Path
from tests.utils.parameters import model_params
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation


@pytest.fixture(scope="session")
def get_local_bucketfs_path() -> str:
    bucket_base_path = ''
    with tempfile.TemporaryDirectory() as tmpdir_name:
        model_path = PurePosixPath(tmpdir_name, bucket_base_path)
        yield str(model_path)


@contextmanager
def download_model(model_name: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir_name:
        for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
            downloader.from_pretrained(model_name, cache_dir=tmpdir_name)
        yield tmpdir_name


@contextmanager
def upload_model(bucketfs_location: AbstractBucketFSLocation,
                 model_name: str, model_dir: str) -> str:

    model_path = bucketfs_operations.get_model_path(
        model_params.sub_dir, model_name)
    bucketfs_operations.upload_model_files_to_bucketfs(
        tmpdir_name=model_dir,
        model_path=Path(model_path),
        bucketfs_location=bucketfs_location)
    yield model_path


@contextmanager
def upload_model_to_local_bucketfs(model_name: str) -> str:
    with download_model(model_name) as download_tmpdir:
        with tempfile.TemporaryDirectory() as upload_tmpdir_name:
            bucketfs_location = LocalFSMockBucketFSLocation(upload_tmpdir_name)
            upload_model(bucketfs_location, model_name, download_tmpdir)
            yield upload_tmpdir_name


@pytest.fixture(scope="session")
def upload_model_base_to_local_bucketfs() -> PurePosixPath:
    with upload_model_to_local_bucketfs(model_params.base) as path:
        yield path


@pytest.fixture(scope="session")
def upload_model_seq2seq_to_local_bucketfs() -> PurePosixPath:
    with upload_model_to_local_bucketfs(model_params.seq2seq) as path:
        yield path


@contextmanager
def upload_model_to_bucketfs(
        model_name: str, bucketfs_location: AbstractBucketFSLocation) -> str:

    with download_model(model_name) as download_tmpdir:
        with upload_model(
                bucketfs_location, model_name, download_tmpdir) as model_path:
            yield model_path


@pytest.fixture(scope="session")
def upload_model_base_to_bucketfs(
        bucketfs_location: AbstractBucketFSLocation) -> PurePosixPath:

    with upload_model_to_bucketfs(
            model_params.base, bucketfs_location) as path:
        try:
            yield path
        finally:
            _cleanup_buckets(bucketfs_location, path)


@pytest.fixture(scope="session")
def upload_model_seq2seq_to_bucketfs(
        bucketfs_location: AbstractBucketFSLocation) -> PurePosixPath:

    with upload_model_to_bucketfs(
            model_params.seq2seq, bucketfs_location) as path:
        try:
            yield path
        finally:
            _cleanup_buckets(bucketfs_location, path)


def _cleanup_buckets(bucketfs_location: AbstractBucketFSLocation, path: str):
    bucketfs_files = bucketfs_location.list_files_in_bucketfs(str(path))
    for file_ in bucketfs_files:
        try:
            bucketfs_location.delete_file_in_bucketfs(
                str(PurePosixPath(path, file_)))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")