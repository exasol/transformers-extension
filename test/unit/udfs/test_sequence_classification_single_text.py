from exasol_transformers_extension.udfs.models.sequence_classification_single_text_udf import \
    SequenceClassificationSingleTextUDF
from test.unit.udf_wrapper_params.sequence_classification.error_on_prediction_single_model_multiple_batch import (
    ErrorOnPredictionSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.sequence_classification.multiple_model_multiple_batch_complete import (
    MultipleModelMultipleBatchComplete,
)

from test.unit.utils.utils_for_udf_tests import create_mock_udf_context, create_mock_exa_environment, \
    create_mock_model_factories_with_models, create_mock_pipeline_factory, assert_correct_number_of_results, \
    assert_result_matches_expected_output
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
            Column("return_rank", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("return_rank", str, "VARCHAR(2000000)"),
            Column("label", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("rank", int, "INTEGER"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize(
    "params",
    [MultipleModelMultipleBatchComplete, ErrorOnPredictionSingleModelMultipleBatch],
)#todo add test cases with differen return rank
# return_ALL_single_model_single_batch
# return_ALL_single_model_multiple_batch
# return_ALL_multiple_model_single_batch
# return_ALL_multiple_model_multiple_batch
# return_HIGHEST_single_model_single_batch
# return_HIGHEST_single_model_multiple_batch
# return_HIGHEST_multiple_model_single_batch
# return_HIGHEST_multiple_model_multiple_batch
# return_mixed_single_model_single_batch #todo more here?
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_sequence_classification_single_text(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path


    model_input_data = params.inputs_single_text
    bfs_connections = params.bfs_connections
    expected_model_counter = params.expected_single_text_model_counter
    sequence_models_output_df = params.sequence_models_output_df_single_text
    batch_size = params.batch_size
    expected_output_data = params.outputs_single_text


    mock_meta = create_mock_metadata()
    mock_ctx = create_mock_udf_context(model_input_data, mock_meta)
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    mock_base_model_factory, mock_tokenizer_factory = (
        create_mock_model_factories_with_models(expected_model_counter)
    )
    mock_pipeline_factory = create_mock_pipeline_factory(
        sequence_models_output_df, expected_model_counter
    )

    udf = SequenceClassificationSingleTextUDF(
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

