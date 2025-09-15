import contextlib
from unittest.mock import (
    Mock,
    call,
)

import exasol.bucketfs as bfs
import pytest
import transformers as huggingface
from _pytest.monkeypatch import MonkeyPatch

import exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
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
    mspec = BucketFSModelSpecification(
        model_name="model name",
        task_type="task type",
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
    assert downloads == [call(huggingface.AutoModel), call(huggingface.AutoTokenizer)]
    assert actual == bfs_location / "some_path"
