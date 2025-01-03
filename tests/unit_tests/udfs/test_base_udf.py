from unittest.mock import patch
import pytest

from tests.unit_tests.utils.utils_for_base_udf_tests import (run_test, create_mock_metadata,
                                                             create_mock_metadata_with_span, regex_matcher)
from tests.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)

from tests.unit_tests.udf_wrapper_params.base_udf.error_not_cached_multiple_model_multiple_batch import \
    ErrorNotCachedMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.error_not_cached_single_model_multiple_batch import \
    ErrorNotCachedSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.base_udf.multiple_bfsconn_single_subdir_single_model_multiple_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch
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

from tests.unit_tests.utils.utils_for_udf_tests import create_mock_udf_context, create_mock_exa_environment, \
    create_mock_pipeline_factory, create_mock_model_factories_with_models, assert_correct_number_of_results, \
    assert_result_matches_expected_output


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

    assert_correct_number_of_results(res, mock_meta.output_columns, expected_output_data)
    assert_result_matches_expected_output(res, expected_output_data,  mock_meta.input_columns)
    #assert len(mock_pipeline_factory.mock_calls) == expected_model_counter

