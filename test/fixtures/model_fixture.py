"""Fixtures for loading standard models to Local BucketFS and DB BucketFS for tests"""

import typing
from pathlib import (
    Path,
    PurePosixPath,
)
from test.fixtures.model_fixture_utils import (
    prepare_default_model_for_local_bucketfs,
    prepare_model_for_local_bucketfs,
    upload_model_to_bucketfs,
    upload_model_to_bucketfs_from_bfs_model_spec,
)
from test.utils.parameters import model_params

import exasol.bucketfs as bfs
import pytest

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)


@pytest.fixture(scope="session")
def prepare_fill_mask_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    """
    Create tmpdir and save standard fill mask model into it, returns tmpdir-path.
    The model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.base_model_specs
    model_specification.task_type = "fill-mask"
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_question_answering_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save standard question answering model into it,
    returns tmpdir-path.
    The model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.q_a_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_text_classification_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save standard text classification model into it,
    returns tmpdir-path.
    The model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.text_classification_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_default_sentiment_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save default sentiment/text classification pair model
    into it, returns tmpdir-path.
    The model is defined in DEFAULT_MODEL_SPECS.
    """
    model_specification = DEFAULT_MODEL_SPECS["AiSentimentUDF"]
    bucketfs_path = prepare_default_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_text_classification_pair_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save default text classification pair model
    into it, returns tmpdir-path.
    Model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.text_classification_pair_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_text_generation_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    """
    Create tmpdir and save standard text generation model into it, returns tmpdir-path.
    Model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.text_gen_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_token_classification_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save standard token classification model into it,
    returns tmpdir-path.
    Model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.token_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_translation_model_for_local_bucketfs(tmpdir_factory) -> PurePosixPath:
    """
    create tmpdir and save standard translation model into it, returns tmpdir-path
    model is defined in test/utils/parameters.py
    """
    model_specification = model_params.seq2seq_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_zero_shot_classification_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save standard zero shot classification model into it,
    returns tmpdir-path.
    Model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.zero_shot_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def prepare_tiny_model_for_local_bucketfs(
    tmpdir_factory,
) -> PurePosixPath:
    """
    Create tmpdir and save tiny model into it, returns tmpdir-path.
    Model is defined in test/utils/parameters.py.
    """
    model_specification = model_params.tiny_model_specs
    bucketfs_path = prepare_model_for_local_bucketfs(
        model_specification, tmpdir_factory
    )
    return bucketfs_path


@pytest.fixture(scope="session")
def upload_fill_mask_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard fill mask model into BucketFS at bucketfs_location,
    returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    base_model_specs = model_params.base_model_specs
    base_model_specs.task_type = "fill-mask"
    tmpdir = tmpdir_factory.mktemp(base_model_specs.task_type)
    with upload_model_to_bucketfs(base_model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_question_answering_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard question answering model into BucketFS at bucketfs_location,
    returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.q_a_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_text_classification_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard text classification model into BucketFS at bucketfs_location,
    returns BucketFS path
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.text_classification_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_default_sentiment_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load default sentiment/text classification model into BucketFS at bucketfs_location,
    returns BucketFS path
    The model is defined in DEFAULT_MODEL_SPECS.
    """
    model_specs = DEFAULT_MODEL_SPECS["AiSentimentUDF"]
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs_from_bfs_model_spec(
        model_specs, tmpdir, bucketfs_location
    ) as path:
        yield path


@pytest.fixture(scope="session")
def upload_text_classification_pair_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard text classification pair model into BucketFS at
    bucketfs_location,
    returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.text_classification_pair_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_text_generation_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard text generation model into BucketFS at bucketfs_location,
    returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.text_gen_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_token_classification_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard token classification model into BucketFS at bucketfs_location,
    returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.token_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_translation_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard translation model into BucketFS at bucketfs_location,
    returns BucketFS path
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.seq2seq_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_zero_shot_classification_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard zero shot classification model into BucketFS at bucketfs_location,
    returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.zero_shot_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_tiny_model_to_bucketfs(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard zero shot classification model into BucketFS at bucketfs_location, returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.tiny_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(model_specs, tmpdir, bucketfs_location) as path:
        yield path


@pytest.fixture(scope="session")
def upload_tiny_model_to_bucketfs_ls_test_subdir(
    bucketfs_location: bfs.path.PathLike, tmpdir_factory
) -> typing.Generator:
    """
    Load standard zero shot classification model into BucketFS at bucketfs_location, returns BucketFS path.
    Model is defined in test/utils/parameters.py.
    """
    model_specs = model_params.tiny_model_specs
    tmpdir = tmpdir_factory.mktemp(model_specs.task_type)
    with upload_model_to_bucketfs(
        model_specs,
        tmpdir,
        bucketfs_location,
        bucketfs_model_subdir=model_params.ls_test_subdir,
    ) as path:
        yield path
