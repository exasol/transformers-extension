import pytest
from pathlib import PurePosixPath

import exasol.bucketfs as bfs

from test.fixtures.model_fixture_utils import prepare_model_for_local_bucketfs, upload_model_to_bucketfs
from test.utils.parameters import model_params


@pytest.fixture(scope="session")
def prepare_filling_mask_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.base_model_specs
    model_specification.task_type = "fill-mask"
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_question_answering_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.q_a_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_sequence_classification_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.sequence_class_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_sequence_classification_pair_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.sequence_class_pair_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_text_generation_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.text_gen_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_token_classification_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.token_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_translation_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.seq2seq_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path

@pytest.fixture(scope="session")
def prepare_zero_shot_classification_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.zero_shot_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path


@pytest.fixture(scope="session")
def prepare_seq2seq_model_in_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.seq2seq_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(model_specification, tmpdir_factory)
    yield bucketfs_path


@pytest.fixture(scope="session")
def upload_filling_mask_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    base_model_specs = model_params.base_model_specs
    base_model_specs.task_type = "fill-mask"
    tmpdir = tmpdir_factory.mktemp(base_model_specs.task_type)
    with upload_model_to_bucketfs(
            base_model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_question_answering_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.q_a_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path

@pytest.fixture(scope="session")
def upload_sequence_classification_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.sequence_class_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path

@pytest.fixture(scope="session")
def upload_sequence_classification_pair_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.sequence_class_pair_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path

@pytest.fixture(scope="session")
def upload_text_generation_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.text_gen_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path

@pytest.fixture(scope="session")
def upload_token_classification_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.token_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path

@pytest.fixture(scope="session")
def upload_translation_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.seq2seq_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path

@pytest.fixture(scope="session")
def upload_zero_shot_classification_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specs = model_params.zero_shot_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
            model_specs, tmpdir, bucketfs_location) as path:
        yield path



@pytest.fixture(scope="session")
def upload_seq2seq_model_to_bucketfs(
        bucketfs_location: bfs.path.PathLike, tmpdir_factory) -> PurePosixPath:
    model_specification = model_params.seq2seq_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specification.task_type)
    with upload_model_to_bucketfs(
            model_specification, tmpdir, bucketfs_location) as path:
        yield path
