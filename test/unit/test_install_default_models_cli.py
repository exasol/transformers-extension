from unittest.mock import (
    Mock,
    call,
)

import transformers as huggingface
from _pytest.monkeypatch import MonkeyPatch
from click.testing import CliRunner
from exasol.pytest_extension import _cli_params_to_args

import exasol_transformers_extension.install_default_models
from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)
from exasol_transformers_extension.install_default_models import (
    install_default_models_command,
)


def test_install_default_models_cli(
    monkeypatch: MonkeyPatch,
):
    cli_args = {
        "bucketfs_host": "bfs-host",
        "bucketfs_port": 2580,
        "bucketfs_name": "bfsdefault",
        "bucket": "default",
        "bucketfs_user": "w",
        "bucketfs_password": "password",
    }
    args_string = _cli_params_to_args(
        cli_args
    )  # we don't want the fixture, just the formated params

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

    runner = CliRunner()
    result = runner.invoke(
        install_default_models_command, args=args_string, catch_exceptions=False
    )
    if result.exit_code != 0:
        print("Exception:", result.exception)
        print("ExcInfo:", result.exc_info)
        print("STDERR:", result.stderr_bytes)
        print("STDOUT:", result.stdout_bytes)
    assert result.exit_code == 0

    expected_bfs_location_call = call(
        bucketfs_host="bfs-host",
        bucketfs_port=2580,
        bucketfs_name="bfsdefault",
        bucket="default",
        bucketfs_user="w",
        bucketfs_password="password",
        bucketfs_use_https=False,
        saas_url="https://cloud.exasol.com",
        saas_account_id=None,
        saas_database_id=None,
        saas_database_name=None,
        saas_token=None,
        path_in_bucket="",
        use_ssl_cert_validation=True,
    )
    expected_model_installs = []
    for udf_name in DEFAULT_MODEL_SPECS:
        expected_model_installs.append(
            call(
                bucketfs_location=bfs_location,
                model_spec=DEFAULT_MODEL_SPECS[udf_name],
                tokenizer_factory=huggingface.AutoTokenizer,
                huggingface_token=None,
            )
        )
    assert create_bfs_mock.call_args == expected_bfs_location_call
    assert install_mock.call_args_list == expected_model_installs
