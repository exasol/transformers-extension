from typing import Union
from unittest.mock import create_autospec, MagicMock, Mock, patch
import re

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from tests.unit_tests.udf_wrapper_params.base_udf.error_not_cached_multiple_model_multiple_batch import \
    ErrorNotCachedMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.error_not_cached_single_model_multiple_batch import \
    ErrorNotCachedSingleModelMultipleBatch
from tests.unit_tests.udfs.test_token_classification import assert_result_matches_expected_output, \
    assert_correct_number_of_results
from tests.unit_tests.utils_for_udf_tests import create_mock_exa_environment, create_mock_udf_context, \
    create_mock_exa_environment_with_token_con, create_base_mock_model_factories, \
    create_mock_model_factories_with_models, create_mock_pipeline_factory
from tests.unit_tests.udfs.base_model_dummy_implementation import DummyImplementationUDF
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from tests.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)


class regex_matcher:
    """Assert that a given string meets some expectations."""
    def __init__(self, pattern, flags=0):
        self._regex = re.compile(pattern, flags)

    def __eq__(self, actual):
        return bool(self._regex.match(actual))

    def __repr__(self):
        return self._regex.pattern


def create_mock_metadata() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
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

def input_for_model_loader_tests(model_name: str,
                                 sub_dir: str,
                                 bucketfs_conn_name: str): # todo move to params file? for consistency?
    input_data = [
        (
            1,
            model_name,
            sub_dir,
            bucketfs_conn_name,
            ''
        )
    ]
    return input_data

def run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory, mock_pipeline, mock_ctx):
    udf = DummyImplementationUDF(exa=mock_exa,
                                 base_model=mock_base_model_factory,
                                 tokenizer=mock_tokenizer_factory,
                                 pipeline=mock_pipeline)
    udf.run(mock_ctx)
    res = mock_ctx.output
    return res

def setup_base_udf_tests_and_run(bfs_connections, input_data, number_of_intended_used_models, tokenizer_models_output_df):
    mock_base_model_factory, mock_tokenizer_factory = create_mock_model_factories_with_models(number_of_intended_used_models)
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    mock_pipeline_factory = create_mock_pipeline_factory(tokenizer_models_output_df, number_of_intended_used_models)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    res = run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory, mock_pipeline_factory, mock_ctx)
    return res, mock_meta

def setup_model_loader_tests_and_run(bucketfs_conn_name, bucketfs_conn, input_data):#todo do we need?
    mock_base_model_factory, mock_tokenizer_factory = create_base_mock_model_factories()
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment_with_token_con(
        [bucketfs_conn_name],
        [bucketfs_conn],
        mock_meta,
        '',
        None)#todo do we nedd empty token con

    mock_pipeline = Mock()#todo dif
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    res = run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory, mock_pipeline, mock_ctx)
    return res, mock_meta

@pytest.mark.parametrize("params", [
    ErrorNotCachedSingleModelMultipleBatch,
    ErrorNotCachedMultipleModelMultipleBatch
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_base_model_udf(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data = params.input_data
    bfs_connections = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    tokenizer_models_output_df = params.tokenizer_models_output_df
    tokenizer_model_output_df_model1 = params.tokenizer_model_output_df_model1
    print(tokenizer_model_output_df_model1)
    print("___________________________")
    #batch_size = params.batch_size
    expected_output_data = params.output_data

    res, mock_meta = setup_base_udf_tests_and_run(bfs_connections, input_data,
                                                  expected_model_counter,
                                                  tokenizer_models_output_df)
    print(res)

    #todo moce these out of token class tests
    assert_correct_number_of_results(res, mock_meta.output_columns, expected_output_data)
    assert_result_matches_expected_output(res, expected_output_data,  mock_meta.input_columns)
    #assert len(mock_pipeline_factory.mock_calls) == expected_model_counter


@pytest.mark.parametrize(["description", "bucketfs_conn_name", "bucketfs_conn",
                         "sub_dir", "model_name"], [
    ("all given", "test_bucketfs_con_name", Connection(address=f"file:///test"),
     "test_subdir", "test_model")
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_model_loader_all_parameters(mock_local_path, mock_create_loc, description,
                                     bucketfs_conn_name, bucketfs_conn, sub_dir, model_name):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data = input_for_model_loader_tests(model_name, sub_dir, bucketfs_conn_name)

    res, mock_meta = setup_model_loader_tests_and_run(bucketfs_conn_name, bucketfs_conn, input_data)
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
     "test_subdir", None),
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_model_loader_missing_parameters(mock_local_path, mock_create_loc, description,
                                             bucketfs_conn_name, bucketfs_conn, sub_dir, model_name):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data = input_for_model_loader_tests(model_name, sub_dir, bucketfs_conn_name)

    res, mock_meta = setup_model_loader_tests_and_run(bucketfs_conn_name, bucketfs_conn, input_data)

    error_field = res[0][-1]
    expected_error = regex_matcher(f".*For each model model_name, bucketfs_conn and sub_dir need to be provided."
                                   f" Found model_name = {model_name}, bucketfs_conn = .*, sub_dir = {sub_dir}.",
                                   flags=re.DOTALL)
    assert error_field == expected_error
    assert error_field is not None and len(res[0]) == len(mock_meta.output_columns)
