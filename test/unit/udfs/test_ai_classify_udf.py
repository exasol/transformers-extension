
from test.unit.udf_wrapper_params.ai_classify.default_values_multiple_batch_complete import \
    DefaultValuesMultipleBatchComplete

from test.unit.utils.utils_for_udf_tests import (
    assert_correct_number_of_results,
    assert_result_matches_expected_output,
    setup_mocks,
)
from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from exasol_transformers_extension.udfs.models.ai_classify_udf import AiClassifyUDF


def create_mock_metadata():
    return MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("candidate_labels", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("candidate_labels", str, "VARCHAR(2000000)"),
            Column("label", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )

@pytest.mark.parametrize("params", [DefaultValuesMultipleBatchComplete])
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch("exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path")
def test_ai_classify_udf(mock_local_path, mock_create_loc, params):
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
        params.inputs,
        params.text_class_models_output_df,
    )

    udf = AiClassifyUDF(
        exa=mock_exa,
        batch_size=params.batch_size,
        base_model=mock_base_model_factory,
        tokenizer=mock_tokenizer_factory,
        pipeline=mock_pipeline_factory,
    )
    udf.run(mock_ctx)
    result = mock_ctx.output

    assert_correct_number_of_results(
        result, mock_meta.output_columns, params.outputs_single_text
    )
    assert_result_matches_expected_output(
        result, params.outputs_single_text, mock_meta.input_columns
    )
    assert len(mock_pipeline_factory.mock_calls) == params.expected_model_counter
