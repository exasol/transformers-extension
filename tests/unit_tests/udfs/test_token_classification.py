from typing import Union, List
from unittest.mock import patch, MagicMock, create_autospec

import pytest
import transformers
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_context import StandaloneMockContext
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from transformers import Pipeline

from exasol_transformers_extension.udfs.models.token_classification_udf import TokenClassificationUDF
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from tests.unit_tests.udf_wrapper_params.token_classification.single_bfsconn_multiple_subdir_single_model_single_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.token_classification.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.token_classification.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete

from tests.unit_tests.udfs.output_matcher import Output, OutputMatcher
from tests.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)
from tests.utils.mock_cast import mock_cast

def udf_wrapper_empty():
    # placeholder to use for MockMetaData creation.
    pass

def create_mock_metadata_with_span():
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper_empty,
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
            Column("text_data_docid", int, "INTEGER"),
            Column("text_data_char_begin", int, "INTEGER"),
            Column("text_data_char_end", int, "INTEGER"),
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

def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper_empty,
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
    #SingleModelMultipleBatchIncomplete,
    #SingleModelMultipleBatchComplete,
    #MultipleModelSingleBatchIncomplete,
    #MultipleModelSingleBatchComplete,
    #MultipleModelMultipleBatchIncomplete,
    #MultipleModelMultipleBatchComplete,
    #MultipleModelMultipleBatchMultipleModelsPerBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    #SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    #MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
    #MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
    #MultipleStrategySingleModelNameSingleBatch,
    #MultipleStrategySingleModelNameMultipleBatch,
    #ErrorNotCachedSingleModelMultipleBatch,
    #ErrorNotCachedMultipleModelMultipleBatch,
    #ErrorOnPredictionMultipleModelMultipleBatch,
    #ErrorOnPredictionSingleModelMultipleBatch
])

@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_token_classification_with_span(mock_local_path, mock_create_loc, params):
    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    mock_meta = create_mock_metadata_with_span()
    input = params.work_with_span_input_data
    mock_ctx = StandaloneMockContext(inp=input, metadata=mock_meta)


    mock_exa = MockExaEnvironment(
        metadata=mock_meta,
        connections=params.bfs_connections)

    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol,
                                                                                      _name="mock_base_model_factory")
    mock_models: List[Union[transformers.AutoModel, MagicMock]] = [
        create_autospec(transformers.AutoModel)
        ]
    mock_cast(mock_base_model_factory.from_pretrained).side_effect = mock_models

    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    mock_pipeline:  List[Union[transformers.AutoModel, MagicMock]] = [
        create_autospec(Pipeline, side_effect=[params.tokenizer_model_output_df])
        ]
    mock_pipeline_factory: Union[Pipeline, MagicMock] = create_autospec(Pipeline,
                                                                        side_effect=mock_pipeline)
    udf = TokenClassificationUDF(exa=mock_exa,
                                 batch_size=params.batch_size,
                                 base_model=mock_base_model_factory,
                                 tokenizer=mock_tokenizer_factory,
                                 pipeline=mock_pipeline_factory,
                                 work_with_spans=True)

    udf.run(mock_ctx)
    result = mock_ctx.output

    assert result[0][-1] is None and len(result[0]) == len(mock_meta.output_columns)

    expected_output = Output(params.work_with_span_output_data)
    actual_output = Output(result)
    n_input_columns = len(mock_meta.input_columns) - 1
    assert (
            OutputMatcher(actual_output, n_input_columns) == expected_output,
            mock_pipeline_factory.mock_calls == params.expected_model_counter)



@pytest.mark.parametrize("params", [
    SingleModelSingleBatchIncomplete,
    SingleModelSingleBatchComplete,
    # SingleModelMultipleBatchIncomplete,
    # SingleModelMultipleBatchComplete,
    # MultipleModelSingleBatchIncomplete,
    # MultipleModelSingleBatchComplete,
    # MultipleModelMultipleBatchIncomplete,
    # MultipleModelMultipleBatchComplete,
    # MultipleModelMultipleBatchMultipleModelsPerBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    # SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    # MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
    # MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
    # MultipleStrategySingleModelNameSingleBatch,
    # MultipleStrategySingleModelNameMultipleBatch,
    # ErrorNotCachedSingleModelMultipleBatch,
    # ErrorNotCachedMultipleModelMultipleBatch,
    # ErrorOnPredictionMultipleModelMultipleBatch,
    # ErrorOnPredictionSingleModelMultipleBatch
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_token_classification(mock_local_path, mock_create_loc, params):
    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    mock_meta = create_mock_metadata()
    input = params.input_data
    mock_ctx = StandaloneMockContext(inp=input, metadata=mock_meta)


    mock_exa = MockExaEnvironment(
        metadata=mock_meta,
        connections=params.bfs_connections)

    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol,
                                                                                      _name="mock_base_model_factory")
    number_of_intendet_used_models = params.expected_model_counter# todo is this always same?
    mock_models: List[Union[transformers.AutoModel, MagicMock]] = [
        create_autospec(transformers.AutoModel) for i in range (0,number_of_intendet_used_models)
        ]
    print(mock_models)
    mock_cast(mock_base_model_factory.from_pretrained).side_effect = mock_models

    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    print(params.batch_size)
    print(params.work_with_span_input_data)
    print(len(params.work_with_span_output_data))
    print("tokenizer_model_output_df")
    print(params.tokenizer_model_output_df)
    mock_pipeline:  List[Union[transformers.AutoModel, MagicMock]] = [
        create_autospec(Pipeline, side_effect=[params.tokenizer_model_output_df[i]]) for i in range (0,number_of_intendet_used_models)
        ]
    mock_pipeline_factory: Union[Pipeline, MagicMock] = create_autospec(Pipeline,
                                                                        side_effect=mock_pipeline)
    udf = TokenClassificationUDF(exa=mock_exa,
                                 batch_size=params.batch_size,
                                 base_model=mock_base_model_factory,
                                 tokenizer=mock_tokenizer_factory,
                                 pipeline=mock_pipeline_factory)

    udf.run(mock_ctx)
    result = mock_ctx.output
    print(result)
    assert result[0][-1] is None and len(result[0]) == len(mock_meta.output_columns)

    expected_output = Output(params.output_data)
    actual_output = Output(result)
    n_input_columns = len(mock_meta.input_columns) - 1
    assert (
            OutputMatcher(actual_output, n_input_columns) == expected_output,
            mock_pipeline_factory.mock_calls == params.expected_model_counter)
