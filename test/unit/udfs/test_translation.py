from exasol_transformers_extension.udfs.models.translation_udf import TranslationUDF
from test.unit.udf_wrapper_params.translation.error_on_prediction_multiple_model_multiple_batch import (
    ErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.translation.error_on_prediction_single_model_multiple_batch import (
    ErrorOnPredictionSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.translation.multiple_language_single_model_multiple_batch import (
    MultipleLanguageSingleModelNameMultipleBatch,
)
from test.unit.udf_wrapper_params.translation.multiple_language_single_model_single_batch import (
    MultipleLanguageSingleModelNameSingleBatch,
)
from test.unit.udf_wrapper_params.translation.multiple_max_length_single_model_multiple_batch import (
    MultipleMaxLengthSingleModelNameMultipleBatch,
)
from test.unit.udf_wrapper_params.translation.multiple_max_length_single_model_single_batch import (
    MultipleMaxLengthSingleModelNameSingleBatch,
)
from test.unit.udf_wrapper_params.translation.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete

from test.unit.utils.utils_for_udf_tests import create_mock_udf_context, create_mock_exa_environment, \
    create_mock_model_factories_with_models, create_mock_pipeline_factory, assert_result_matches_expected_output, \
    assert_correct_number_of_results
from test.utils.mock_bucketfs_location import (
    fake_bucketfs_location_from_conn_object,
    fake_local_bucketfs_path,
)
from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("source_language", str, "VARCHAR(2000000)"),
            Column("target_language", str, "VARCHAR(2000000)"),
            Column("max_length", int, "INTEGER"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("source_language", str, "VARCHAR(2000000)"),
            Column("target_language", str, "VARCHAR(2000000)"),
            Column("max_length", int, "INTEGER"),
            Column("translation_text", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize(
    "params",
    [
        MultipleMaxLengthSingleModelNameSingleBatch,
        MultipleMaxLengthSingleModelNameMultipleBatch,
        MultipleLanguageSingleModelNameSingleBatch,
        MultipleLanguageSingleModelNameMultipleBatch,
        MultipleModelMultipleBatchComplete,
        ErrorOnPredictionMultipleModelMultipleBatch,
        ErrorOnPredictionSingleModelMultipleBatch,
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_translation(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    model_input_data = params.input_data
    bfs_connection = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    translation_models_output_df = params.translation_models_output_df
    batch_size = params.batch_size
    expected_output_data = params.output_data

    mock_meta = create_mock_metadata()
    mock_ctx = create_mock_udf_context(input_data=model_input_data, mock_meta=mock_meta)
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connection)
    mock_base_model_factory, mock_tokenizer_factory = (
        create_mock_model_factories_with_models(expected_model_counter)
    )
    mock_pipeline_factory = create_mock_pipeline_factory(
        translation_models_output_df, expected_model_counter
    )

    udf = TranslationUDF(
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

