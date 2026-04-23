from test.unit.udf_wrapper_params.ai_classify_extended.return_ALL_error_on_prediction_multiple_model_multiple_batch import (
    ReturnAllErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_ALL_error_on_prediction_single_model_multiple_batch import (
    ReturnAllErrorOnPredictionSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_ALL_multiple_labels_single_model_multiple_batch import (
    ReturnAllMultipleLabelsSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_ALL_multiple_labels_single_model_single_batch import (
    ReturnAllMultipleLabelsSingleModelSingleBatch,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_ALL_multiple_model_multiple_batch_complete import (
    ReturnAllMultipleModelMultipleBatchComplete,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_HIGHEST_error_on_prediction_multiple_model_multiple_batch import (
    ReturnHighestErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_HIGHEST_multiple_model_multiple_batch_complete import (
    ReturnHighestMultipleModelMultipleBatchComplete,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_mixed_error_on_prediction_multiple_model_multiple_batch import (
    ReturnMixedErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_classify_extended.return_mixed_multiple_model_multiple_batch import (
    ReturnMixedMultipleModelMultipleBatchComplete,
)
from test.unit.utils.utils_for_udf_tests import (
    assert_correct_number_of_results,
    assert_result_matches_expected_output,
    setup_mocks,
)

from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol_transformers_extension.udfs.models.ai_classify_extended_udf import (
    AiClassifyExtendedUDF,
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
            Column("candidate_labels", str, "VARCHAR(2000000)"),
            Column("return_ranks", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data_doc_id", int, "INTEGER"),
            Column("text_data_char_begin", int, "INTEGER"),
            Column("text_data_char_end", int, "INTEGER"),
            Column("return_ranks", str, "VARCHAR(2000000)"),
            Column("label", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("rank", int, "INTEGER"),
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
            Column("candidate_labels", str, "VARCHAR(2000000)"),
            Column("return_ranks", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("candidate_labels", str, "VARCHAR(2000000)"),
            Column("return_ranks", str, "VARCHAR(2000000)"),
            Column("label", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("rank", int, "INTEGER"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize(
    "params",
    [
        ReturnAllMultipleModelMultipleBatchComplete,
        ReturnAllMultipleLabelsSingleModelSingleBatch,
        ReturnAllMultipleLabelsSingleModelMultipleBatch,
        ReturnAllErrorOnPredictionMultipleModelMultipleBatch,
        ReturnAllErrorOnPredictionSingleModelMultipleBatch,
        ReturnHighestErrorOnPredictionMultipleModelMultipleBatch,
        ReturnHighestMultipleModelMultipleBatchComplete,
        ReturnMixedErrorOnPredictionMultipleModelMultipleBatch,
        ReturnMixedMultipleModelMultipleBatchComplete,
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_ai_classify_extended(mock_local_path, mock_create_loc, params):
    """
    This test checks combinations of input data to determine correct output data. For this everything the udf uses in
    the background is mocked, and given to the udf. we then check if the resulting output matches the expected output.
    """
    mock_meta = create_mock_metadata()
    expected_model_counter = params.expected_model_counter
    batch_size = params.batch_size
    expected_output_data = params.output_data

    (mock_exa, mock_base_model_factory, mock_tokenizer_factory,
     mock_pipeline_factory, mock_ctx) = setup_mocks(
        mock_create_loc, mock_local_path,
        params, mock_meta, expected_model_counter,
        params.input_data,
        params.zero_shot_models_output_df
    )

    udf = AiClassifyExtendedUDF(
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


@pytest.mark.parametrize(
    "params",
    [
        ReturnAllMultipleModelMultipleBatchComplete,
        ReturnAllMultipleLabelsSingleModelSingleBatch,
        ReturnAllMultipleLabelsSingleModelMultipleBatch,
        ReturnAllErrorOnPredictionMultipleModelMultipleBatch,
        ReturnAllErrorOnPredictionSingleModelMultipleBatch,
        ReturnHighestErrorOnPredictionMultipleModelMultipleBatch,
        ReturnHighestMultipleModelMultipleBatchComplete,
        ReturnMixedErrorOnPredictionMultipleModelMultipleBatch,
        ReturnMixedMultipleModelMultipleBatchComplete,
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_ai_classify_extended_with_span(mock_local_path, mock_create_loc, params):
    """
    This test checks combinations of input data to determine correct output data. For this everything the udf uses in
    the background is mocked, and given to the udf. we then check if the resulting output matches the expected output.
    """

    mock_meta = create_mock_metadata_with_span()
    expected_model_counter = params.expected_model_counter
    batch_size = params.batch_size
    expected_output_data = params.work_with_span_output_data

    (mock_exa, mock_base_model_factory, mock_tokenizer_factory,
     mock_pipeline_factory, mock_ctx) = setup_mocks(
        mock_create_loc, mock_local_path,
        params, mock_meta, expected_model_counter,
        params.work_with_span_input_data,
        params.zero_shot_models_output_df
    )

    udf = AiClassifyExtendedUDF(
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
