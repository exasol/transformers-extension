
from unittest.mock import (
    Mock,
    call,
)

import transformers as huggingface
from _pytest.monkeypatch import MonkeyPatch

import exasol_transformers_extension.install_default_models
from exasol_transformers_extension.install_default_models import install_default_models
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)
from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_MODEL_SPECS



def test_install_default_models(monkeypatch: MonkeyPatch, capsys):
    bfs_location = Mock()
    create_bfs_mock = Mock(return_value=bfs_location)
    monkeypatch.setattr(
        exasol_transformers_extension.install_default_models,
        "create_bucketfs_location",
        create_bfs_mock,
    )
    install_mock = Mock(return_value="Some path")
    monkeypatch.setattr(
        exasol_transformers_extension.install_default_models,
        "install_huggingface_model",
        install_mock,
    )

    install_default_models(
        bucketfs_host="bfs-host",
        bucketfs_port=2580,
        bucketfs_name="bfsdefault",
        bucket="default",
        bucketfs_user="w",
        bucketfs_password="password",
        token="my token",
    )

    expected_model_installs = []
    for mspec in DEFAULT_MODEL_SPECS:
        expected_model_installs.append(call(
            bucketfs_location=bfs_location,
            model_spec=mspec,
            tokenizer_factory=huggingface.AutoTokenizer,
            huggingface_token=None
        ))
    assert install_mock.call_args_list == expected_model_installs
    captured = capsys.readouterr()
    assert captured.out.count("A model or tokenizer has been saved in the BucketFS at: ") == len(DEFAULT_MODEL_SPECS)

