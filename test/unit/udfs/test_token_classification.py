# test params:
from test.unit.udf_wrapper_params.token_classification.error_on_prediction_multiple_model_multiple_batch import (
    ErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.error_on_prediction_single_model_multiple_batch import (
    ErrorOnPredictionSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.error_prediction_containing_only_unknown_fields import (
    ErrorPredictionOnlyContainsUnknownFields,
)
from test.unit.udf_wrapper_params.token_classification.error_prediction_missing_expected_field import (
    ErrorPredictionMissingExpectedFields,
)
from test.unit.udf_wrapper_params.token_classification.multiple_model_multiple_batch_complete_multiple_entities import (
    MultipleModelMultipleBatchCompleteMultipleEntities,
)
from test.unit.udf_wrapper_params.token_classification.multiple_strategy_single_model_multiple_batch import (
    MultipleStrategySingleModelNameMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.multiple_strategy_single_model_single_batch import (
    MultipleStrategySingleModelNameSingleBatch,
)
from test.unit.udf_wrapper_params.token_classification.prediction_contains_additional_keys import (
    PredictionContainsAdditionalFields,
)
from test.unit.udf_wrapper_params.token_classification.prediction_returns_empty_result import (
    PredictionReturnsEmptyResult,
)
from test.unit.utils.utils_for_udf_tests import (
    assert_correct_number_of_results,
    assert_result_matches_expected_output,
    create_mock_exa_environment,
    create_mock_model_factories_with_models,
    create_mock_pipeline_factory,
    create_mock_udf_context,
)
from test.utils.mock_bucketfs_location import (
    fake_bucketfs_location_from_conn_object,
    fake_local_bucketfs_path,
)
from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol_transformers_extension.udfs.models.token_classification_udf import (
    TokenClassificationUDF,
)


def create_mock_metadata_with_span():
    """Creates mock metadata for UDF tests. includes span columns"""
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("text_data_doc_id", int, "INTEGER"),
            Column("text_data_char_begin", int, "INTEGER"),
            Column("text_data_char_end", int, "INTEGER"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data_doc_id", int, "INTEGER"),
            Column("text_data_char_begin", int, "INTEGER"),
            Column("text_data_char_end", int, "INTEGER"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)"),
            Column("entity_covered_text", str, "VARCHAR(2000000)"),
            Column("entity_type", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("entity_doc_id", int, "INTEGER"),
            Column("entity_char_begin", int, "INTEGER"),
            Column("entity_char_end", int, "INTEGER"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


def create_mock_metadata():
    """Creates mock metadata for UDF tests"""
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("aggregation_strategy", str, "VARCHAR(2000000)"),
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
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize(
    "params",
    [
        MultipleStrategySingleModelNameSingleBatch,
        MultipleStrategySingleModelNameMultipleBatch,
        ErrorOnPredictionMultipleModelMultipleBatch,
        ErrorOnPredictionSingleModelMultipleBatch,
        PredictionReturnsEmptyResult,
        ErrorPredictionMissingExpectedFields,
        ErrorPredictionOnlyContainsUnknownFields,
        PredictionContainsAdditionalFields,
        MultipleModelMultipleBatchCompleteMultipleEntities,
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_token_classification_with_span(mock_local_path, mock_create_loc, params):
    """
    This test checks combinations of input data to determine correct output data. For this everything the udf uses in
    the background is mocked, and given to the udf. we then check if the resulting output matches the expected output.
    """
    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    model_input_data = params.work_with_span_input_data
    bfs_connections = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    tokenizer_models_output_df = params.tokenizer_models_output_df
    batch_size = params.batch_size
    expected_output_data = params.work_with_span_output_data

    mock_meta = create_mock_metadata_with_span()
    mock_ctx = create_mock_udf_context(model_input_data, mock_meta)
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    mock_base_model_factory, mock_tokenizer_factory = (
        create_mock_model_factories_with_models(expected_model_counter)
    )
    mock_pipeline_factory = create_mock_pipeline_factory(
        tokenizer_models_output_df, expected_model_counter
    )

    udf = TokenClassificationUDF(
        exa=mock_exa,
        batch_size=batch_size,
        base_model=mock_base_model_factory,
        tokenizer=mock_tokenizer_factory,
        pipeline=mock_pipeline_factory,
        work_with_spans=True,
    )

    udf.run(mock_ctx)
    result = mock_ctx.output

    assert_correct_number_of_results(
        result, mock_meta.output_columns, expected_output_data
    )
    assert_result_matches_expected_output(
        result, expected_output_data, mock_meta.input_columns
    )
    assert len(mock_pipeline_factory.mock_calls) == expected_model_counter


@pytest.mark.parametrize(
    "params",
    [
        MultipleStrategySingleModelNameSingleBatch,
        MultipleStrategySingleModelNameMultipleBatch,
        ErrorOnPredictionMultipleModelMultipleBatch,
        ErrorOnPredictionSingleModelMultipleBatch,
        PredictionReturnsEmptyResult,
        ErrorPredictionMissingExpectedFields,
        ErrorPredictionOnlyContainsUnknownFields,
        PredictionContainsAdditionalFields,
        MultipleModelMultipleBatchCompleteMultipleEntities,
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_token_classification(mock_local_path, mock_create_loc, params):
    """
    This test checks combinations of input data to determine correct output data. For this everything the udf uses in
    the background is mocked, and given to the udf. we then check if the resulting output matches the expected output.
    """
    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    model_input_data = params.input_data
    bfs_connections = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    tokenizer_models_output_df = params.tokenizer_models_output_df
    batch_size = params.batch_size
    expected_output_data = params.output_data

    mock_meta = create_mock_metadata()
    mock_ctx = create_mock_udf_context(model_input_data, mock_meta)
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    mock_base_model_factory, mock_tokenizer_factory = (
        create_mock_model_factories_with_models(expected_model_counter)
    )
    mock_pipeline_factory = create_mock_pipeline_factory(
        tokenizer_models_output_df, expected_model_counter
    )

    udf = TokenClassificationUDF(
        exa=mock_exa,
        batch_size=batch_size,
        base_model=mock_base_model_factory,
        tokenizer=mock_tokenizer_factory,
        pipeline=mock_pipeline_factory,
    )

    udf.run(mock_ctx)
    result = mock_ctx.output

    assert_correct_number_of_results(
        result, mock_meta.output_columns, expected_output_data
    )
    assert_result_matches_expected_output(
        result, expected_output_data, mock_meta.input_columns
    )
    assert len(mock_pipeline_factory.mock_calls) == expected_model_counter
