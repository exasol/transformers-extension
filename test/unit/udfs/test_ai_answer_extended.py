from test.unit.udf_wrapper_params.ai_answer_extended.error_on_prediction_multiple_model_multiple_batch import (
    ErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_answer_extended.error_on_prediction_single_model_multiple_batch import (
    ErrorOnPredictionSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.ai_answer_extended.multiple_model_multiple_batch_complete import (
    MultipleModelMultipleBatchComplete,
)
from test.unit.udfs.output_matcher import (
    Output,
    OutputMatcher,
)
from test.unit.utils.utils_for_udf_tests import (
    assert_correct_number_of_results,
    setup_mocks,
)
from test.utils.mock_bucketfs_location import (
    fake_bucketfs_location_from_conn_object,
    fake_local_bucketfs_path,
)
from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol_transformers_extension.udfs.models.ai_answer_extended_udf import (
    AiAnswerExtendedUDF,
)


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("question", str, "VARCHAR(2000000)"),
            Column("context_text", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("question", str, "VARCHAR(2000000)"),
            Column("context_text", str, "VARCHAR(2000000)"),
            Column("answer", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize(
    "params",
    [
        MultipleModelMultipleBatchComplete,
        ErrorOnPredictionSingleModelMultipleBatch,
        ErrorOnPredictionMultipleModelMultipleBatch,
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_ai_answer_extended(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    mock_meta = create_mock_metadata()

    (
        mock_exa,
        mock_base_model_factory,
        mock_tokenizer_factory,
        mock_pipeline_factory,
        mock_ctx,
    ) = setup_mocks(
        mock_create_loc,
        mock_local_path,
        params,
        mock_meta,
        params.expected_model_counter,
        params.input_data,
        params.models_output_df,
    )

    udf = AiAnswerExtendedUDF(
        exa=mock_exa,
        batch_size=params.batch_size,
        base_model=mock_base_model_factory,
        tokenizer=mock_tokenizer_factory,
        pipeline=mock_pipeline_factory,
    )

    udf.run(mock_ctx)
    result = mock_ctx.output
    result_output = Output(result)
    expected_output = Output(params.expected_output_data)
    n_input_columns = len(mock_meta.input_columns) - 1

    assert_correct_number_of_results(
        result, mock_meta.output_columns, params.expected_output_data
    )

    assert (
        OutputMatcher(result_output, n_input_columns) == expected_output
        and len(mock_pipeline_factory.mock_calls) == params.expected_model_counter
    )
