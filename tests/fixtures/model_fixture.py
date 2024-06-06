import pytest
import transformers
from contextlib import contextmanager
from pathlib import PurePosixPath, Path

from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString
from tests.utils import postprocessing
from tests.utils.parameters import model_params
from exasol_transformers_extension.utils import bucketfs_operations
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_bucketfs_utils_python.abstract_bucketfs_location import \
    AbstractBucketFSLocation


def download_model_to_standard_local_save_path(model_specification_string: ModelSpecificationString,
                                               tmpdir_name: Path) -> Path:
    tmpdir_name = Path(tmpdir_name)
    local_model_save_path = bucketfs_operations.create_save_pretrained_model_path(tmpdir_name,
                                                                                  model_specification_string)
    model_name = model_specification_string.deconstruct()
    for model_factory in [transformers.AutoModel, transformers.AutoTokenizer]:
        model = model_factory.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        model.save_pretrained(local_model_save_path)
    return local_model_save_path


def download_model_to_path(model_specification_string: ModelSpecificationString,
                           tmpdir_name: Path):
    tmpdir_name = Path(tmpdir_name)
    model_name = model_specification_string.deconstruct()
    # todo pull this download into a function? -> create ticket
    for model_factory in [transformers.AutoModel, transformers.AutoTokenizer]:
        model = model_factory.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        model.save_pretrained(tmpdir_name)


@contextmanager
def upload_model(bucketfs_location: AbstractBucketFSLocation,
                 model_specification_string: ModelSpecificationString,
                 model_dir: Path) -> Path:
    model_path = bucketfs_operations.get_bucketfs_model_save_path(
        model_params.sub_dir, model_specification_string)
    bucketfs_operations.upload_model_files_to_bucketfs(
        model_directory=str(model_dir),
        bucketfs_model_path=Path(model_path),
        bucketfs_location=bucketfs_location)
    yield model_path


def prepare_model_for_local_bucketfs(model_specification_string: ModelSpecificationString, tmpdir_factory):
    model_name = model_specification_string.deconstruct()
    tmpdir = tmpdir_factory.mktemp(model_name)
    model_path_in_bucketfs = bucketfs_operations.get_bucketfs_model_save_path(model_params.sub_dir,
                                                                              model_specification_string)
    bucketfs_path_for_model = tmpdir / model_path_in_bucketfs
    download_model_to_path(model_specification_string, bucketfs_path_for_model)
    return tmpdir


@pytest.fixture(scope="session")
def prepare_base_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification_string = ModelSpecificationString(model_params.base_model)
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification_string, tmpdir_factory)
    yield bucketfs_path


@pytest.fixture(scope="session")
def prepare_seq2seq_model_in_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification_string = ModelSpecificationString(model_params.seq2seq_model)
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification_string, tmpdir_factory)
    yield bucketfs_path


@contextmanager
def upload_model_to_bucketfs(
        model_specification_string: ModelSpecificationString,
        download_tmpdir: Path,
        bucketfs_location: AbstractBucketFSLocation) -> str:
    download_tmpdir = download_model_to_standard_local_save_path(model_specification_string, download_tmpdir)
    with upload_model(
            bucketfs_location, model_specification_string, download_tmpdir) as model_path:
        try:
            yield model_path
        finally:
            postprocessing.cleanup_buckets(bucketfs_location, str(model_path.parent))


@pytest.fixture(scope="session")
def upload_base_model_to_bucketfs(
        bucketfs_location, tmpdir_factory) -> PurePosixPath:
    tmpdir = tmpdir_factory.mktemp(model_params.base_model)
    with upload_model_to_bucketfs(
            ModelSpecificationString(model_params.base_model), tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_seq2seq_model_to_bucketfs(
        bucketfs_location, tmpdir_factory) -> PurePosixPath:
    tmpdir = tmpdir_factory.mktemp(model_params.seq2seq_model)
    with upload_model_to_bucketfs(
            ModelSpecificationString(model_params.seq2seq_model), tmpdir, bucketfs_location) as path:
        yield path
