import time
from contextlib import contextmanager
from pathlib import Path

import transformers
from exasol import bucketfs as bfs

from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.utils.bucketfs_model_specification import \
    get_BucketFSModelSpecification_from_model_Specs, BucketFSModelSpecification
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    make_parameters_of_model_contiguous_tensors
from exasol_transformers_extension.utils.model_specification import ModelSpecification
from tests.utils import postprocessing
from tests.utils.parameters import model_params


def download_model_to_standard_local_save_path(model_specification: ModelSpecification,
                                               tmpdir_name: Path) -> Path:
    tmpdir_name = Path(tmpdir_name)
    local_model_save_path = bucketfs_operations.create_save_pretrained_model_path(tmpdir_name,
                                                                                  model_specification)
    model_name = model_specification.model_name
    model_factory = model_specification.get_model_factory()
    for model in [model_factory, transformers.AutoTokenizer]:
        downloaded_model = model.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        make_parameters_of_model_contiguous_tensors(downloaded_model)
        downloaded_model.save_pretrained(local_model_save_path)
    return local_model_save_path


def download_model_to_path(model_specification: ModelSpecification,
                           tmpdir_name: Path):
    tmpdir_name = Path(tmpdir_name)
    model_name = model_specification.model_name
    model_factory = model_specification.get_model_factory()
    for model in [model_factory, transformers.AutoTokenizer]:
        downloaded_model = model.from_pretrained(model_name, cache_dir=tmpdir_name / "cache" / model_name)
        make_parameters_of_model_contiguous_tensors(downloaded_model)
        downloaded_model.save_pretrained(tmpdir_name)

def prepare_model_for_local_bucketfs(model_specification: ModelSpecification,
                                     tmpdir_factory):
    current_model_specs = get_BucketFSModelSpecification_from_model_Specs(model_specification,
                                                                          "",
                                                                          model_params.sub_dir)

    tmpdir = tmpdir_factory.mktemp(current_model_specs.task_type)
    model_path_in_bucketfs = current_model_specs.get_bucketfs_model_save_path()

    bucketfs_path_for_model = tmpdir / model_path_in_bucketfs
    download_model_to_path(current_model_specs, bucketfs_path_for_model)
    return tmpdir


@contextmanager
def upload_model_to_bucketfs(
        model_specification: ModelSpecification,
        local_model_save_path: Path,
        bucketfs_location: bfs.path.PathLike) -> str:
    local_model_save_path = download_model_to_standard_local_save_path(model_specification, local_model_save_path)
    current_model_specs = get_BucketFSModelSpecification_from_model_Specs(model_specification,
                                                                          "",
                                                                          model_params.sub_dir)
    with upload_model(
            bucketfs_location, current_model_specs, local_model_save_path) as model_path:
        try:
            yield model_path
        finally:
            postprocessing.cleanup_buckets(bucketfs_location, model_path)


@contextmanager
def upload_model(bucketfs_location: bfs.path.PathLike,
                 current_model_specification: BucketFSModelSpecification,
                 model_dir: Path) -> Path:
    model_path = current_model_specification.get_bucketfs_model_save_path()
    bucketfs_operations.upload_model_files_to_bucketfs(
        model_directory=str(model_dir),
        bucketfs_model_path=Path(model_path),
        bucketfs_location=bucketfs_location)
    time.sleep(20)
    yield model_path
