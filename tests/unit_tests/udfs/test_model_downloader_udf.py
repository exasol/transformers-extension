from pathlib import PosixPath
from typing import Union, List
from unittest.mock import create_autospec, MagicMock, call, Mock, patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from tests.unit_tests.utils_for_udf_tests import create_mock_exa_environment, create_mock_udf_context
from exasol_transformers_extension.udfs.models.model_downloader_udf import \
    ModelDownloaderUDF
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory, HuggingFaceHubBucketFSModelTransferSP
from tests.utils.matchers import AnyOrder
from tests.utils.mock_cast import mock_cast


def create_mock_metadata() -> MockMetaData:
    def udf_wrapper():
        pass

    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_path_in_udfs", str, "VARCHAR(2000000)"),
            Column("model_path_of_tar_file_in_bucketfs", str, "VARCHAR(2000000)")
        ]
    )
    return meta


@pytest.mark.parametrize("count", list(range(1, 10)))
@pytest.mark.parametrize("description, token_conn_name ,token_conn_obj, expected_token", [
    ('without token', '', None, False),
    ('with token', 'conn_name', Connection(address="", password="valid"), "valid"),
])
@patch('exasol_transformers_extension.utils.bucketfs_operations.create_bucketfs_location_from_conn_object')
def test_model_downloader(mock_create_loc, description, count, token_conn_name, token_conn_obj, expected_token):

    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    mock_model_downloader_factory: Union[HuggingFaceHubBucketFSModelTransferSPFactory, MagicMock] = create_autospec(
        HuggingFaceHubBucketFSModelTransferSPFactory)
    mock_model_downloaders: List[Union[HuggingFaceHubBucketFSModelTransferSP, MagicMock]] = [
        create_autospec(HuggingFaceHubBucketFSModelTransferSP)
        for i in range(count)]
    for i in range(count):
        mock_cast(mock_model_downloaders[i].__enter__).side_effect = [mock_model_downloaders[i]]
    mock_cast(mock_model_downloader_factory.create).side_effect = mock_model_downloaders
    mock_bucketfs_locations = [Mock() for i in range(count)]
    mock_create_loc.side_effect = mock_bucketfs_locations
    base_model_names = [f"base_model_name_{i}" for i in range(count)]
    sub_directory_names = [f"sub_dir_{i}" for i in range(count)]
    bucketfs_connections = [Connection(address=f"file:///test{i}") for i in range(count)]
    bfs_conn_name = [f"bfs_conn_name_{i}" for i in bucketfs_connections]
    input_data = [
        (
            base_model_names[i],
            sub_directory_names[i],
            bfs_conn_name[i],
            token_conn_name
        )
        for i in range(count)
    ]
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(
        bfs_conn_name,
        bucketfs_connections,
        mock_meta,
        token_conn_name,
        token_conn_obj)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)

    udf = ModelDownloaderUDF(exa=mock_exa,
                             base_model_factory=mock_base_model_factory,
                             tokenizer_factory=mock_tokenizer_factory,
                             huggingface_hub_bucketfs_model_transfer=mock_model_downloader_factory)
    udf.run(mock_ctx)

    assert mock_cast(mock_model_downloader_factory.create).mock_calls == [
        call(bucketfs_location=mock_bucketfs_locations[i],
             model_name=base_model_names[i],
             model_path=PosixPath(f'{sub_directory_names[i]}/{base_model_names[i]}'),
             token=expected_token)
        for i in range(count)
    ]
    for i in range(count):
        assert mock_cast(mock_model_downloaders[i].download_from_huggingface_hub).mock_calls == [
            call(mock_base_model_factory),
            call(mock_tokenizer_factory)
        ]
        assert call() in mock_cast(mock_model_downloaders[i].upload_to_bucketfs).mock_calls
    called_loc_addresses = [arg_list[0][0].address for arg_list in mock_create_loc.call_args_list]
    expected_loc_addresses = [f'file:///test{i}' for i in range(count)]
    assert expected_loc_addresses == AnyOrder(called_loc_addresses)
    assert mock_ctx.output == [
        (
            f'{sub_directory_names[i]}/{base_model_names[i]}',
            str(mock_model_downloaders[i].upload_to_bucketfs())
        )
        for i in range(count)
    ]
