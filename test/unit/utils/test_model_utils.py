import contextlib
from pathlib import Path
from unittest.mock import (
    Mock,
    call,
)

import exasol.bucketfs as bfs
import pytest
import transformers as huggingface
from _pytest.monkeypatch import MonkeyPatch

import exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp
from exasol_transformers_extension.utils import device_management
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.model_utils import (
    install_huggingface_model,
    load_huggingface_pipeline,
)


@pytest.fixture
def bfs_location() -> bfs.path.PathLike:
    return bfs.path.BucketPath("root", bucket_api=Mock())


def test_install_huggingface_model(
    monkeypatch: MonkeyPatch,
    bfs_location: bfs.path.PathLike,
) -> None:
    downloader_mock = Mock()
    downloader_mock.upload_to_bucketfs.return_value = "some_path"

    @contextlib.contextmanager
    def context_mock(**kwargs):
        yield downloader_mock

    monkeypatch.setattr(
        exasol_transformers_extension.utils.model_utils,
        "HuggingFaceHubBucketFSModelTransferSP",
        context_mock,
    )
    mspec = BucketFSModelSpecification(
        model_name="model name",
        task_type="fill-mask",
        bucketfs_conn_name="",
        sub_dir="sub-dir",
    )
    actual = install_huggingface_model(
        bucketfs_location=bfs_location,
        model_spec=mspec,
        tokenizer_factory=huggingface.AutoTokenizer,
        huggingface_token="hf-token",
    )
    downloads = downloader_mock.download_from_huggingface_hub.call_args_list
    assert downloads == [call(huggingface.AutoModelForMaskedLM), call(huggingface.AutoTokenizer)]
    assert actual == bfs_location / "some_path"


def test_load_huggingface_pipeline(monkeypatch: MonkeyPatch):
    model_spec = BucketFSModelSpecification(
        model_name="name",
        task_type="fill-mask",
        bucketfs_conn_name="bfs conn",
        sub_dir=Path("sub"),
    )
    model_mock = Mock()
    loader_mock = Mock()
    loader_mock.load_models.return_value = model_mock

    bfs_mock = Mock()
    get_bucketfs_location_mock = Mock(return_value=bfs_mock)
    monkeypatch.setattr(
        exasol_transformers_extension.utils.model_utils,
        "get_bucketfs_location",
        get_bucketfs_location_mock,
    )
    loader_constructor = Mock(return_value=loader_mock)
    monkeypatch.setattr(
        exasol_transformers_extension.utils.model_utils,
        "LoadLocalModel",
        loader_constructor,
    )
    actual = load_huggingface_pipeline(
        exa=Mock(),
        model_spec=model_spec,
        device=-1,
    )
    expected_device = device_management.get_torch_device(-1)
    assert loader_constructor.call_args == call(
        pipeline_factory=huggingface.pipeline,
        base_model_factory=model_spec.get_model_factory(),
        tokenizer_factory=huggingface.AutoTokenizer,
        task_type=model_spec.task_type,
        device=expected_device,
    )
    loader_mock.clear_device_memory.called
    loader_mock.set_current_model_specification.call_args = call(model_spec)
    loader_mock.set_bucketfs_model_cache_dir.call_args = call(bfs_mock)
    assert actual == model_mock
