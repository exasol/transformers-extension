from test.unit.utils.utils_for_udf_tests import (
    create_mock_exa_environment_with_token_con,
    create_mock_udf_context,
)
from unittest.mock import (
    Mock,
    call,
    patch,
)

import pytest
from _pytest.monkeypatch import MonkeyPatch
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

import exasol_transformers_extension
from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
    DEFAULT_MODEL_SPECS,
    DEFAULT_SUBDIR,
)
from exasol_transformers_extension.udfs.models.install_default_models_udf import (
    InstallDefaultModelsUDF,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
)


def create_mock_metadata() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[],
        output_type="EMITS",
        output_columns=[
            Column("model_path_in_udfs", str, "VARCHAR(2000000)"),
            Column("model_path_of_tar_file_in_bucketfs", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
def test_model_downloader(mock_create_loc, monkeypatch: MonkeyPatch):

    mock_bucketfs_location = Mock()
    mock_create_loc.side_effect = mock_bucketfs_location

    bucketfs_connection = [Connection(address=f"file:///test")]
    bucketfs_conn_name = [DEFAULT_BUCKETFS_CONN_NAME]

    mock_return_path = []
    for udf_name in DEFAULT_MODEL_SPECS:
        model_name = DEFAULT_MODEL_SPECS[udf_name].model_name
        mock_return_path.append(
            (f"{DEFAULT_SUBDIR}/{model_name}", model_name + "_tar_fiel_path")
        )

    download_model_mock = Mock(side_effect=mock_return_path)
    monkeypatch.setattr(
        exasol_transformers_extension.utils.in_udf_model_downloader.InUDFModelDownloader,
        "download_model",
        download_model_mock,
    )

    input_data = []
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment_with_token_con(
        bucketfs_conn_name,
        bucketfs_connection,
        mock_meta,
        None,
        None,
    )
    mock_ctx = create_mock_udf_context(input_data, mock_meta)

    udf = InstallDefaultModelsUDF(
        exa=mock_exa,
    )
    udf.run(mock_ctx)
    expected_download_model_calls = [
        call(None, DEFAULT_MODEL_SPECS[udf_name], mock_exa)
        for udf_name in DEFAULT_MODEL_SPECS
    ]

    assert expected_download_model_calls == download_model_mock.call_args_list
    assert mock_ctx.output == mock_return_path
