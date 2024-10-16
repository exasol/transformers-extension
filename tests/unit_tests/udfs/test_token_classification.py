from unittest.mock import patch, Mock

import pytest
import transformers
from exasol_udf_mock_python.mock_context import StandaloneMockContext
from transformers.pipelines import Pipeline
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from typing import Union, List
from unittest.mock import create_autospec, MagicMock, Mock, patch
import re

import pytest
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from tests.utils.mock_cast import mock_cast
from tests.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)


from exasol_transformers_extension.udfs.models.token_classification_udf import TokenClassificationUDF
from tests.unit_tests.udf_wrapper_params.token_classification.error_not_cached_multiple_model_multiple_batch import \
    ErrorNotCachedMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.error_not_cached_single_model_multiple_batch import \
    ErrorNotCachedSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.error_on_prediction_multiple_model_multiple_batch import \
    ErrorOnPredictionMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.error_on_prediction_single_model_multiple_batch import \
    ErrorOnPredictionSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_bfsconn_single_subdir_single_model_multiple_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_bfsconn_single_subdir_single_model_single_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_model_multiple_batch_incomplete import \
    MultipleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_model_multiple_batch_multiple_models_per_batch import \
    MultipleModelMultipleBatchMultipleModelsPerBatch
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_model_single_batch_complete import \
    MultipleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_model_single_batch_incomplete import \
    MultipleModelSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_strategy_single_model_multiple_batch import \
    MultipleStrategySingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.multiple_strategy_single_model_single_batch import \
    MultipleStrategySingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.single_bfsconn_multiple_subdir_single_model_multiple_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.single_bfsconn_multiple_subdir_single_model_single_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.single_model_multiple_batch_complete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.token_classification.single_model_multiple_batch_incomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.token_classification.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.token_classification.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete
from tests.unit_tests.udfs.output_matcher import Output, OutputMatcher
from tests.unit_tests.utils_for_udf_tests import create_mock_udf_context
from tests.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)


def create_mock_metadata_with_span(udf_wrapper):
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("text_data_docid", int, "INTEGER"),
            Column("text_data_char_begin", int, "INTEGER"),
            Column("text_data_char_end", int, "INTEGER"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)"),
            Column("entity_covered_text", str, "VARCHAR(2000000)"),
            Column("entity_type", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("entity_docid", int, "INTEGER"),
            Column("entity_char_begin", int, "INTEGER"),
            Column("entity_char_end", int, "INTEGER"),
            Column("error_message", str, "VARCHAR(2000000)")
        ],
    )
    return meta

def create_mock_metadata(udf_wrapper):
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)"),
            Column("start_pos", int, "INTEGER"),
            Column("end_pos", int, "INTEGER"),
            Column("word", str, "VARCHAR(2000000)"),
            Column("entity", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("error_message", str, "VARCHAR(2000000)")
        ],
    )
    return meta

@pytest.mark.parametrize("params", [
    SingleModelSingleBatchIncomplete,
    SingleModelSingleBatchComplete,
    #todo are we ok with changing the tests like i did in these two cases?
    # if yes i will add the changes to all the other param files as well
    #SingleModelMultipleBatchIncomplete,
   # SingleModelMultipleBatchComplete,
   # MultipleModelSingleBatchIncomplete,
   # MultipleModelSingleBatchComplete,
    #MultipleModelMultipleBatchIncomplete,
    #MultipleModelMultipleBatchComplete,
    #MultipleModelMultipleBatchMultipleModelsPerBatch,
    #SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    #SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    #MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
    #MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
   # MultipleStrategySingleModelNameSingleBatch,
    #MultipleStrategySingleModelNameMultipleBatch,
    #ErrorNotCachedSingleModelMultipleBatch,
    #ErrorNotCachedMultipleModelMultipleBatch,
    #ErrorOnPredictionMultipleModelMultipleBatch,
    #ErrorOnPredictionSingleModelMultipleBatch
])
@patch('exasol_transformers_extension.utils.bucketfs_operations.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_token_classification_with_span(mock_local_path, mock_create_loc, params):
    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    mock_meta = create_mock_metadata_with_span(params.work_with_span_udf_wrapper)
    input = params.work_with_span_input_data
    mock_ctx = StandaloneMockContext(inp=input, metadata=mock_meta)


    mock_exa = MockExaEnvironment(
        metadata=mock_meta,
        connections=params.bfs_connections)

    #--------
    # do we want to still use this?
    #executor = UDFMockExecutor()
    #result = executor.run(mock_ctx, mock_exa) # todo this would need  changes in UDFMockExecutor
    #--------
    # or do we want a version like below?
    # or is there a secret magic power the STandAloneUDFMock has that i am not seeing?
    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol, _name="mock_base_model_factory")
    mock_models: List[Union[transformers.AutoModel, MagicMock]] = [
        create_autospec(transformers.AutoModel)
        ]
    mock_cast(mock_base_model_factory.from_pretrained).side_effect = mock_models

    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    mock_pipeline:  List[Union[transformers.AutoModel, MagicMock]] = [
        create_autospec(Pipeline, side_effect=[params.tokenizer_model_output_df])
        ]
    mock_pipeline_factory: Union[Pipeline, MagicMock] = create_autospec(Pipeline, side_effect=mock_pipeline) #todo do we want to use params.mock_pipeline instead?
    # todo do we want to use the udf or the mock in params?
    udf = TokenClassificationUDF(exa=mock_exa,
                                 base_model=mock_base_model_factory,
                                 tokenizer=mock_tokenizer_factory,
                                 pipeline=mock_pipeline_factory,
                                 work_with_spans=True)

    udf.run(mock_ctx)
    result = mock_ctx.output

    assert result[0][-1] is None and len(result[0]) == len(mock_meta.output_columns)

    expected_output = mock_meta.output_columns
    n_input_columns = len(mock_meta.input_columns) - 1
    try:
        assert (
            OutputMatcher(result, n_input_columns) == expected_output, #validate emit checks output type but not content, but output matcher does not work with the result from mock_ctx
            params.mock_pipeline.counter == params.expected_model_counter) # pipeline counter not incresed in mock
    finally:
        params.mock_pipeline.counter = 0



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
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
    MultipleStrategySingleModelNameSingleBatch,
    MultipleStrategySingleModelNameMultipleBatch,
    ErrorNotCachedSingleModelMultipleBatch,
    ErrorNotCachedMultipleModelMultipleBatch,
    ErrorOnPredictionMultipleModelMultipleBatch,
    ErrorOnPredictionSingleModelMultipleBatch
])
@patch('exasol_transformers_extension.utils.bucketfs_operations.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_token_classification(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)

    exa = MockExaEnvironment(
        metadata=meta,
        connections=params.bfs_connections)

    result = executor.run([Group(params.input_data)], exa)
    result_output = Output(result[0].rows)
    expected_output = Output(params.output_data)
    n_input_columns = len(meta.input_columns) - 1
    try:
        assert (
            OutputMatcher(result_output, n_input_columns) == expected_output,
            params.mock_pipeline.counter == params.expected_model_counter)
    finally:
        params.mock_pipeline.counter = 0
