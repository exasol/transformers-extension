import re
from pathlib import Path
from unittest.mock import (
    Mock,
    call,
)

import transformers as huggingface
from _pytest.monkeypatch import MonkeyPatch

import exasol_transformers_extension.upload_model
from exasol_transformers_extension.upload_model import upload_model
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)


def test_upload_model(monkeypatch: MonkeyPatch, capsys):
    bfs_location = Mock()
    create_bfs_mock = Mock(return_value=bfs_location)
    monkeypatch.setattr(
        exasol_transformers_extension.upload_model,
        "create_bucketfs_location",
        create_bfs_mock,
    )
    install_mock = Mock(return_value="Some path")
    monkeypatch.setattr(
        exasol_transformers_extension.upload_model,
        "install_huggingface_model",
        install_mock,
    )
    mspec = BucketFSModelSpecification(
        "model name",
        "task type",
        "",
        Path("sub dir"),
    )
    upload_model(
        model_name=mspec.model_name,
        task_type=mspec.task_type,
        sub_dir=str(mspec.sub_dir),
        bucketfs_host="bfs-host",
        bucketfs_port=2580,
        bucketfs_name="bfsdefault",
        bucket="default",
        bucketfs_user="w",
        bucketfs_password="password",
        token="my token",
    )
    assert install_mock.call_args == call(
        bucketfs_location=bfs_location,
        model_spec=mspec,
        tokenizer_factory=huggingface.AutoTokenizer,
        huggingface_token="my token",
    )
    captured = capsys.readouterr()
    assert re.match(r"Your model .* has been saved .* at: Some path", captured.out)
