from pathlib import PosixPath, Path
from typing import Union, Any, Tuple, List
from unittest.mock import create_autospec, MagicMock, call, Mock

import pytest
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification, \
    CurrentModelSpecificationFactory
from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString
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
def test_model_downloader(description, count, token_conn_name, token_conn_obj, expected_token):
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

    mock_bucketfs_factory: Union[BucketFSFactory, MagicMock] = create_autospec(BucketFSFactory)
    mock_bucketfs_locations = [Mock() for i in range(count)]
    mock_cast(mock_bucketfs_factory.create_bucketfs_location).side_effect = mock_bucketfs_locations
    base_model_names = [f"base_model_name_{i}" for i in range(count)]
    sub_directory_names = [f"sub_dir_{i}" for i in range(count)]
    bucketfs_connections = [Connection(address=f"file:///test{i}") for i in range(count)]
    bfs_conn_name = [f"bfs_conn_name_{i}" for i in bucketfs_connections]

    mock_cmss = [create_autospec(CurrentModelSpecification,
                                 model_name=base_model_names[i],
                                 sub_dir=Path(sub_directory_names[i])) for i in range(count)]
    for i in range(count):
        mock_cast(mock_cmss[i].get_bucketfs_model_save_path).side_effect = [f'{sub_directory_names[i]}/{base_model_names[i]}']
    mock_current_model_specification_string_factory: Union[CurrentModelSpecificationFactory, MagicMock] = (
        create_autospec(CurrentModelSpecificationFactory))
    mock_cast(mock_current_model_specification_string_factory.create).side_effect = mock_cmss

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
                             huggingface_hub_bucketfs_model_transfer=mock_model_downloader_factory,
                             bucketfs_factory=mock_bucketfs_factory,
                             current_model_specification_string_factory=mock_current_model_specification_string_factory)
    udf.run(mock_ctx)
    assert mock_cast(mock_model_downloader_factory.create).mock_calls == [
        call(bucketfs_location=mock_bucketfs_locations[i],
             model_specification_string=mock_cmss[i],
             model_path=f'{sub_directory_names[i]}/{base_model_names[i]}',
             token=expected_token)
        for i in range(count)
    ]
    for i in range(count):
        assert mock_cast(mock_model_downloaders[i].download_from_huggingface_hub).mock_calls == [
            call(mock_base_model_factory),
            call(mock_tokenizer_factory)
        ]
        assert call() in mock_cast(mock_model_downloaders[i].upload_to_bucketfs).mock_calls
    assert mock_cast(mock_bucketfs_factory.create_bucketfs_location).mock_calls == AnyOrder([
        call(url=f'file:///test{i}', user=None, pwd=None)
        for i in range(count)
    ])
    assert mock_ctx.output == [
        (
            f'{sub_directory_names[i]}/{base_model_names[i]}',
            str(mock_model_downloaders[i].upload_to_bucketfs())
        )
        for i in range(count)
    ]
