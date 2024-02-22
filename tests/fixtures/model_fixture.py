import pytest
import transformers
from contextlib import contextmanager
from pathlib import PurePosixPath, Path
from tests.utils import postprocessing
from tests.utils.parameters import model_params
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation


def download_model(model_name: str, tmpdir_name: Path) -> Path:
    tmpdir_name = Path(tmpdir_name)
    for model_factory in [transformers.AutoModel, transformers.AutoTokenizer]:
        model = model_factory.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        model.save_pretrained(tmpdir_name / "pretrained" / model_name)
    return tmpdir_name / "pretrained" / model_name


@contextmanager
def upload_model(bucketfs_location: AbstractBucketFSLocation,
                 model_name: str, model_dir: Path) -> Path:
    model_path = bucketfs_operations.get_model_path(
        model_params.sub_dir, model_name)
    bucketfs_operations.upload_model_files_to_bucketfs(
        tmpdir_name=str(model_dir),
        model_path=Path(model_path),
        bucketfs_location=bucketfs_location)
    yield model_path


def generate_local_bucketfs_path_for_model(tmpdir, model):
    return tmpdir / model_params.sub_dir / model.replace("-", "_")


def prepare_model_in_local_bucketfs(model: str, tmpdir_factory):
    yield from prepare_model_in_local_bucketfs(model, tmpdir_factory)
    tmpdir = tmpdir_factory.mktemp(model)
    bucketfs_path_for_model = generate_local_bucketfs_path_for_model(tmpdir, model)
    download_model(model, bucketfs_path_for_model)
    yield tmpdir


@pytest.fixture(scope="session")
def prepare_base_model_in_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model = model_params.base_model
    yield from prepare_model_in_local_bucketfs(model, tmpdir_factory)


@pytest.fixture(scope="session")
def prepare_seq2seq_model_in_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model = model_params.seq2seq_model
    yield from prepare_model_in_local_bucketfs(model, tmpdir_factory)


@contextmanager
def upload_model_to_bucketfs(
        model_name: str,
        download_tmpdir: Path,
        bucketfs_location: AbstractBucketFSLocation) -> str:
    download_model(model_name, download_tmpdir)
    with upload_model(
            bucketfs_location, model_name, download_tmpdir) as model_path:
        try:
            yield model_path
        finally:
            postprocessing.cleanup_buckets(bucketfs_location, str(model_path.parent))


@pytest.fixture(scope="session")
def upload_base_model_to_bucketfs(
        bucketfs_location, tmpdir_factory) -> PurePosixPath:
    tmpdir = tmpdir_factory.mktemp(model_params.base_model)
    with upload_model_to_bucketfs(
            model_params.base_model, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_seq2seq_model_to_bucketfs(
        bucketfs_location, tmpdir_factory) -> PurePosixPath:
    tmpdir = tmpdir_factory.mktemp(model_params.seq2seq_model)
    with upload_model_to_bucketfs(
            model_params.seq2seq_model, tmpdir, bucketfs_location) as path:
        yield path
