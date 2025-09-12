import contextlib
from unittest.mock import (
    Mock,
    call,
)

import exasol.bucketfs as bfs
import pytest
from _pytest.monkeypatch import MonkeyPatch
from transformers import AutoTokenizer

import exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp
from exasol_transformers_extension.utils.model_utils import (
    install_huggingface_model,
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
    model_factory = Mock()
    actual = install_huggingface_model(
        bucketfs_location=bfs_location,
        sub_dir="sub-dir",
        task_type="task type",
        model_name="model name",
        model_factory=model_factory,
        tokenizer_factory=AutoTokenizer,
        huggingface_token="hf-token",
    )
    downloads = downloader_mock.download_from_huggingface_hub.call_args_list
    assert downloads == [call(model_factory), call(AutoTokenizer)]
    assert actual == bfs_location / "some_path"
