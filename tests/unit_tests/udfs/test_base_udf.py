from typing import Union, Any, Tuple, List
from unittest.mock import create_autospec, MagicMock, call, Mock

import pytest
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from tests.unit_tests.utils_for_udf_tests import create_mock_exa_environment, create_mock_udf_context
from tests.unit_tests.udfs.base_model_dummy_implementation import DummyImplementationUDF
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer import ModelFactoryProtocol
from tests.utils.mock_cast import mock_cast
import re


class regex_matcher: #this one feels more correct
    """Assert that a given string meets some expectations."""
    def __init__(self, pattern, flags=0):
        self._regex = re.compile(pattern, flags)

    def is_in(self, actual):
        return bool(self._regex.search(actual))

    def __repr__(self):
        return self._regex.pattern


class regex_matcher2: #this one looks nicer when used and gives better error messages
    """Assert that a given string meets some expectations."""
    def __init__(self, some_sting):
        self._string = some_sting

    def __contains__(self, pattern):
        return bool(re.search(pattern, self._string))

    def __repr__(self):
        return self._string


def create_mock_metadata() -> MockMetaData:
    def udf_wrapper():
        pass

    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
            Column("answer", bool, "BOOLEAN"),
            Column("score", str, "DOUBLE"),
            Column("error_message", str, "VARCHAR(2000000)")
        ]
    )
    return meta


def test_setup(description, bucketfs_conn_name, bucketfs_conn, sub_dir, model_name):
    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)

    mock_bucketfs_factory: Union[BucketFSFactory, MagicMock] = create_autospec(BucketFSFactory)
    mock_bucketfs_locations = [Mock()]
    mock_cast(mock_bucketfs_factory.create_bucketfs_location).side_effect = mock_bucketfs_locations

    input_data = [
        (
            1,
            model_name,
            sub_dir,
            bucketfs_conn_name,
            ''
        )
    ]
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(
        [bucketfs_conn_name],
        [bucketfs_conn],
        mock_meta,
        '',
        None)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    udf = DummyImplementationUDF(exa=mock_exa,
                             base_model=mock_base_model_factory,
                             tokenizer=mock_tokenizer_factory)
    udf.run(mock_ctx)
    res = mock_ctx.output
    return res, mock_meta



@pytest.mark.parametrize(["description", "bucketfs_conn_name", "bucketfs_conn",
                         "sub_dir", "model_name"], [
    ("all given", "test_bucketfs_con_name", Connection(address=f"file:///test"),
     "test_subdir", "test_model")
])
def test_model_downloader_all_parameters(description, bucketfs_conn_name, bucketfs_conn, sub_dir, model_name):

    res, mock_meta = test_setup(description, bucketfs_conn_name, bucketfs_conn, sub_dir, model_name)
    # check if no errors
    assert res[0][-1] is None and len(res[0]) == len(mock_meta.output_columns)


@pytest.mark.parametrize(["description", "bucketfs_conn_name", "bucketfs_conn",
                         "sub_dir", "model_name"], [
    ("all null", None, None, None, None),
    ("model name missing", "test_bucketfs_con_name", Connection(address=f"file:///test"),
     "test_subdir", None),
    ("bucketfs_conn missing", None, None,
     "test_subdir", "test_model"),
    ("sub_dir missing", "test_bucketfs_con_name", Connection(address=f"file:///test"),
     None, "test_model"),
    ("model_name missing", "test_bucketfs_con_name", Connection(address=f"file:///test"),
     "test_subdir", None)
])
def test_model_downloader_missing_parameters(description, bucketfs_conn_name, bucketfs_conn, sub_dir, model_name):
    res, mock_meta = test_setup(description, bucketfs_conn_name, bucketfs_conn, sub_dir, model_name)

    error_field = res[0][-1]
    expected_error = regex_matcher(f"For each model model_name, bucketfs_conn and sub_dir need to be provided."
                                   f" Found model_name = {model_name}, bucketfs_conn = .*, sub_dir = {sub_dir}.")
    assert expected_error.is_in(error_field)

    expected_error2 =f"For each model model_name, bucketfs_conn and sub_dir need to be provided. " \
                     f"Found model_name = {model_name}, bucketfs_conn = .*, sub_dir = {sub_dir}."
    error_field_matcher = regex_matcher2(error_field)
    assert expected_error2 in error_field_matcher

    assert error_field is not None and len(res[0]) == len(mock_meta.output_columns)
