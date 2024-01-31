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
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory


def download_model(model_name: str, tmpdir_name: Path) -> None:
    with HuggingFaceHubBucketFSModelTransferSPFactory().create(
            bucketfs_location=,#todo
            model_name=model_name,
            model_path=tmpdir_name,
            token=""
    ) as downloader:
        downloader.download_from_huggingface_hub(model_factory=ModelFactoryProtocol[transformers.AutoModel])
        downloader.download_from_huggingface_hub(model_factory=ModelFactoryProtocol[transformers.AutoTokenizer])



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


@contextmanager
def upload_model_to_local_bucketfs(
        model_name: str, download_tmpdir: Path) -> str:

    download_model(model_name, download_tmpdir)
    upload_tmpdir_name = Path(download_tmpdir, "upload_tmpdir")
    upload_tmpdir_name.mkdir(parents=True, exist_ok=True)
    bucketfs_location = LocalFSMockBucketFSLocation(
        PurePosixPath(upload_tmpdir_name))
    upload_model(bucketfs_location, model_name, download_tmpdir)
    yield upload_tmpdir_name


@pytest.fixture(scope="session")
def upload_base_model_to_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    tmpdir = tmpdir_factory.mktemp(model_params.base_model)
    with upload_model_to_local_bucketfs(
            model_params.base_model, tmpdir) as path:
        yield path


@pytest.fixture(scope="session")
def upload_seq2seq_model_to_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    tmpdir = tmpdir_factory.mktemp(model_params.seq2seq_model)
    with upload_model_to_local_bucketfs(
            model_params.seq2seq_model, tmpdir) as path:
        yield path


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
