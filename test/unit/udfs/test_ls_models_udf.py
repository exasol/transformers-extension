from test.unit.utils.utils_for_base_udf_tests import (
    create_mock_metadata,
    create_mock_metadata_with_span,
    run_test,
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

def create_mock_metadata():
    """Creates mock metadata for UDF tests"""
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("version", str, "VARCHAR(2000000)"),
            Column("task_name", str, "VARCHAR(2000000)"),
            Column("seed", str, "VARCHAR(2000000)"),
            Column("path", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta

def setup_base_udf_tests_and_run(
    bfs_connections,
    input_data,
):

    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    #res = run_test(
    #    mock_exa,
    #    mock_ctx,
    #)
    udf = DummyImplementationUDF(
        exa=mock_exa,
        base_model=mock_base_model_factory,
        batch_size=batch_size,
        tokenizer=mock_tokenizer_factory,
        pipeline=mock_pipeline,
        work_with_spans=work_with_span,
    )
    udf.run(mock_ctx)
    res = mock_ctx.output
    return res, mock_meta


)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_base_model_udf(mock_local_path, mock_create_loc, params):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data = params.input_data
    bfs_connections = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    tokenizer_models_output_df = params.tokenizer_models_output_df

    batch_size = params.batch_size
    expected_output_data = params.output_data

    res, mock_meta, mock_pipeline_factory = setup_base_udf_tests_and_run(
        bfs_connections,
        input_data,
        expected_model_counter,
        tokenizer_models_output_df,
        batch_size,
    )

    assert_correct_number_of_results(
        res, mock_meta.output_columns, expected_output_data
    )
    assert_result_matches_expected_output(
        res, expected_output_data, mock_meta.input_columns
    )
    assert len(mock_pipeline_factory.mock_calls) == expected_model_counter


