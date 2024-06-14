import pytest
import transformers
from contextlib import contextmanager
from pathlib import PurePosixPath, Path

from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification, \
    CurrentModelSpecificationFromModelSpecs
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
    model_name = model_specification_string.model_name
    for model_factory in [transformers.AutoModel, transformers.AutoTokenizer]:
        model = model_factory.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        model.save_pretrained(local_model_save_path)
    return local_model_save_path


def download_model_to_path(model_specification_string: ModelSpecificationString,
                           tmpdir_name: Path):
    tmpdir_name = Path(tmpdir_name)
    model_name = model_specification_string.model_name
    # todo pull this download into a function? -> create ticket
    for model_factory in [transformers.AutoModel, transformers.AutoTokenizer]:
        model = model_factory.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        model.save_pretrained(tmpdir_name)


@contextmanager
def upload_model(bucketfs_location: AbstractBucketFSLocation,
                 current_model_specification_string: CurrentModelSpecification,
                 model_dir: Path) -> Path:
    model_path = current_model_specification_string.get_bucketfs_model_save_path()
    bucketfs_operations.upload_model_files_to_bucketfs(
        model_directory=str(model_dir),
        bucketfs_model_path=Path(model_path),
        bucketfs_location=bucketfs_location)
    yield model_path


def prepare_model_for_local_bucketfs(model_specification_string: ModelSpecificationString,
                                     tmpdir_factory):
    current_model_specs = CurrentModelSpecificationFromModelSpecs().transform(model_specification_string,
                                                                              "",
                                                                              model_params.sub_dir)
    tmpdir = tmpdir_factory.mktemp(current_model_specs.get_model_specific_path_suffix())
    model_path_in_bucketfs = current_model_specs.get_bucketfs_model_save_path()

    bucketfs_path_for_model = tmpdir / model_path_in_bucketfs
    download_model_to_path(current_model_specs, bucketfs_path_for_model)
    return tmpdir


@pytest.fixture(scope="session")
def prepare_base_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification_string = model_params.base_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification_string, tmpdir_factory)
    yield bucketfs_path


@pytest.fixture(scope="session")
def prepare_seq2seq_model_in_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification_string = model_params.seq2seq_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification_string, tmpdir_factory)
    yield bucketfs_path


@contextmanager
def upload_model_to_bucketfs(
        model_specification_string: ModelSpecificationString,
        download_tmpdir: Path,
        bucketfs_location: AbstractBucketFSLocation) -> str:
    download_tmpdir = download_model_to_standard_local_save_path(model_specification_string, download_tmpdir)
    current_model_specs = CurrentModelSpecificationFromModelSpecs().transform(model_specification_string,
                                                                              "",
                                                                              model_params.sub_dir)
    with upload_model(
            bucketfs_location, current_model_specs, download_tmpdir) as model_path:
        try:
            yield model_path
        finally:
            postprocessing.cleanup_buckets(bucketfs_location, str(model_path.parent))


@pytest.fixture(scope="session")
def upload_base_model_to_bucketfs(
        bucketfs_location, tmpdir_factory) -> PurePosixPath:
    base_model_specs = model_params.base_model_specs
    tmpdir = tmpdir_factory.mktemp(base_model_specs.get_model_specific_path_suffix())
    with upload_model_to_bucketfs(
            base_model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_seq2seq_model_to_bucketfs(
        bucketfs_location, tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.seq2seq_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specification.get_model_specific_path_suffix())
    with upload_model_to_bucketfs(
            model_specification, tmpdir, bucketfs_location) as path:
        yield path
