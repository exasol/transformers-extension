from pathlib import Path
from test.unit.utils.utils_for_udf_tests import (
    create_mock_exa_environment_with_token_con,
    create_mock_udf_context,
)
from test.utils.matchers import AnyOrder
from test.utils.mock_cast import mock_cast
from typing import (
    Union,
)
from unittest.mock import (
    MagicMock,
    Mock,
    call,
    create_autospec,
    patch,
)

import pytest
from _pytest.monkeypatch import MonkeyPatch
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

import exasol_transformers_extension
from exasol_transformers_extension.udfs.models.install_default_models_udf import (
    InstallDefaultModelsUDF,
)
from exasol_transformers_extension.udfs.models.model_downloader_udf import (
    ModelDownloaderUDF,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
    BucketFSModelSpecificationFactory,
)
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSP,
)


def create_mock_metadata() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("task_type", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_path_in_udfs", str, "VARCHAR(2000000)"),
            Column("model_path_of_tar_file_in_bucketfs", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize("count", list(range(1, 10)))
@pytest.mark.parametrize(
    "description, token_conn_name ,token_conn_obj, expected_token",
    [
        ("without token", "", None, None),
        ("with token", "conn_name", Connection(address="", password="valid"), "valid"),
    ],
)
def test_model_downloader(
    description,
    count,
    token_conn_name,
    token_conn_obj,
    expected_token,
    monkeypatch: MonkeyPatch,
):
    base_model_names = [f"base_model_name_{i}" for i in range(count)]
    sub_directory_names = [f"sub_dir_{i}" for i in range(count)]
    task_type = "fill_mask"
    bucketfs_connections = [
        Connection(address=f"file:///test{i}") for i in range(count)
    ]

    mock_return_paths = [
        (
            f"{sub_directory_names[i]}/{base_model_names[i]}",
            base_model_names[i] + "_tar_fiel_path",
        )
        for i in range(count)
    ]
    download_model_mock = Mock(side_effect=mock_return_paths)

    monkeypatch.setattr(
        exasol_transformers_extension.utils.in_udf_model_downloader.InUDFModelDownloader,
        "download_model",
        download_model_mock,
    )

    bucketfs_conn_name = [f"bucketfs_conn_name_{i}" for i in bucketfs_connections]

    mock_bucketfs_model_specs = [
        create_autospec(
            BucketFSModelSpecification,
            model_name=base_model_names[i],
            task_type=task_type,
            sub_dir=Path(sub_directory_names[i]),
            bucketfs_conn_name=bucketfs_conn_name[i],
        )
        for i in range(count)
    ]
    for i in range(count):
        mock_cast(
            mock_bucketfs_model_specs[i].get_bucketfs_model_save_path
        ).side_effect = [f"{sub_directory_names[i]}/{base_model_names[i]}"]
    mock_current_model_specification_factory: Union[
        BucketFSModelSpecificationFactory, MagicMock
    ] = create_autospec(BucketFSModelSpecificationFactory)
    mock_cast(mock_current_model_specification_factory.create).side_effect = (
        mock_bucketfs_model_specs
    )

    input_data = [
        (
            base_model_names[i],
            sub_directory_names[i],
            task_type,
            bucketfs_conn_name[i],
            token_conn_name,
        )
        for i in range(count)
    ]
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment_with_token_con(
        bucketfs_conn_name,
        bucketfs_connections,
        mock_meta,
        token_conn_name,
        token_conn_obj,
    )
    mock_ctx = create_mock_udf_context(input_data, mock_meta)

    udf = ModelDownloaderUDF(
        exa=mock_exa,
        current_model_specification_factory=mock_current_model_specification_factory,
    )
    udf.run(mock_ctx)
    expected_download_model_calls = [
        call(token_conn_name, mock_bucketfs_model_specs[i], mock_exa)
        for i in range(count)
    ]
    print(mock_ctx.output)
    assert mock_ctx.output == mock_return_paths
    assert expected_download_model_calls == download_model_mock.call_args_list
