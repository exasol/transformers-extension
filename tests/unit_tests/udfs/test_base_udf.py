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
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_bfsconn_single_subdir_single_model_multiple_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_bfsconn_single_subdir_single_model_single_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_model_multiple_batch_incomplete import \
    MultipleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_model_multiple_batch_multiple_models_per_batch import \
    MultipleModelMultipleBatchMultipleModelsPerBatch
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_model_single_batch_complete import \
    MultipleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_model_single_batch_incomplete import \
    MultipleModelSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.base_udf.single_bfsconn_multiple_subdir_single_model_multiple_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.single_bfsconn_multiple_subdir_single_model_single_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.single_model_multiple_batch_complete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.base_udf.single_model_multiple_batch_incomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.base_udf.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.base_udf.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete
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
            Column("input_data", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),
            Column("answer", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("error_message", str, "VARCHAR(2000000)")
        ]
    )
    return meta

def create_mock_metadata_with_span() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),
            Column("test_span_column_drop", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),#todo add to drop?
            Column("answer", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("test_span_column_add", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ]
    )
    return meta

def data_for_model_loader_tests(model_name: str,
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
    output_data = [
        {
            "aswer": "answer",
            "score": 1.0

        }
    ]
    return input_data, output_data

def run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory,
             mock_pipeline, mock_ctx, batch_size=100, work_with_span=False):
    udf = DummyImplementationUDF(exa=mock_exa,
                                 base_model=mock_base_model_factory,
                                 batch_size=batch_size,
                                 tokenizer=mock_tokenizer_factory,
                                 pipeline=mock_pipeline,
                                 work_with_spans=work_with_span)
    udf.run(mock_ctx)
    res = mock_ctx.output
    return res

def setup_base_udf_tests_and_run(bfs_connections, input_data,
                                 number_of_intended_used_models, tokenizer_models_output_df,
                                 batch_size, work_with_span=False):
    mock_base_model_factory, mock_tokenizer_factory = create_mock_model_factories_with_models(number_of_intended_used_models)
    if work_with_span:
        mock_meta = create_mock_metadata_with_span()
    else:
        mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    mock_pipeline_factory = create_mock_pipeline_factory(tokenizer_models_output_df, number_of_intended_used_models)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    res = run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory,
                   mock_pipeline_factory, mock_ctx, batch_size, work_with_span)
    return res, mock_meta

def setup_model_loader_tests_and_run(bucketfs_conn_name, bucketfs_conn,
                                     input_data, model_output_data):#todo do we need?
    mock_base_model_factory, mock_tokenizer_factory = create_base_mock_model_factories()
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment_with_token_con(
        [bucketfs_conn_name],
        [bucketfs_conn],
        mock_meta,
        '',
        None)#todo do we nedd empty token con

    mock_pipeline_factory = create_mock_pipeline_factory([[[model_output_data]]],
                                                         1)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    res = run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory, mock_pipeline_factory, mock_ctx)
    return res, mock_meta


@pytest.mark.parametrize("params", [
    SingleModelSingleBatchIncomplete,
    SingleModelSingleBatchComplete,
    SingleModelMultipleBatchIncomplete,
    SingleModelMultipleBatchComplete,
    MultipleModelSingleBatchIncomplete,
    MultipleModelSingleBatchComplete,
    MultipleModelMultipleBatchIncomplete,
    MultipleModelMultipleBatchComplete,
    MultipleModelMultipleBatchMultipleModelsPerBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
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

    batch_size = params.batch_size
    expected_output_data = params.output_data

    res, mock_meta = setup_base_udf_tests_and_run(bfs_connections, input_data,
                                                  expected_model_counter,
                                                  tokenizer_models_output_df,
                                                  batch_size)
    print(res)

    #todo moce these out of token class tests
    assert_correct_number_of_results(res, mock_meta.output_columns, expected_output_data)
    assert_result_matches_expected_output(res, expected_output_data,  mock_meta.input_columns)
    #assert len(mock_pipeline_factory.mock_calls) == expected_model_counter

@pytest.mark.parametrize("params", [
    SingleModelSingleBatchIncomplete,
    SingleModelSingleBatchComplete,
    SingleModelMultipleBatchIncomplete,
    SingleModelMultipleBatchComplete,
    MultipleModelSingleBatchIncomplete,
    MultipleModelSingleBatchComplete,
    MultipleModelMultipleBatchIncomplete,
    MultipleModelMultipleBatchComplete,
    MultipleModelMultipleBatchMultipleModelsPerBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
    ErrorNotCachedSingleModelMultipleBatch,
    ErrorNotCachedMultipleModelMultipleBatch
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_base_model_udf_with_span(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data = params.work_with_span_input_data
    bfs_connections = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    tokenizer_models_output_df = params.tokenizer_models_output_df

    batch_size = params.batch_size
    expected_output_data = params.work_with_span_output_data
    work_with_span = True

    res, mock_meta = setup_base_udf_tests_and_run(bfs_connections, input_data,
                                                  expected_model_counter,
                                                  tokenizer_models_output_df,
                                                  batch_size, work_with_span)
    print(res)
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

    input_data, model_output_data = data_for_model_loader_tests(model_name, sub_dir, bucketfs_conn_name)

    res, mock_meta = setup_model_loader_tests_and_run(bucketfs_conn_name, bucketfs_conn, input_data, model_output_data)
    # check if no errors
    assert res[0][-1] is None and len(res[0]) == len(mock_meta.output_columns)


@pytest.mark.parametrize(["description", "bucketfs_conn_name", "bucketfs_conn",
                         "sub_dir", "model_name"], [
    ("all null", None, None, None, None),#todo do we want to change these to use the same format as the moved ones?
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

    input_data, model_output_data = data_for_model_loader_tests(model_name, sub_dir, bucketfs_conn_name)

    res, mock_meta = setup_model_loader_tests_and_run(bucketfs_conn_name, bucketfs_conn, input_data, model_output_data)

    error_field = res[0][-1]
    expected_error = regex_matcher(f".*For each model model_name, bucketfs_conn and sub_dir need to be provided."
                                   f" Found model_name = {model_name}, bucketfs_conn = .*, sub_dir = {sub_dir}.",
                                   flags=re.DOTALL)
    assert error_field == expected_error
    assert error_field is not None and len(res[0]) == len(mock_meta.output_columns)
