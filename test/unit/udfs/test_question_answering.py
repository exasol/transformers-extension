from unittest.mock import patch

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from test.unit.udf_wrapper_params.question_answering.error_on_prediction_multiple_model_multiple_batch import \
    ErrorOnPredictionMultipleModelMultipleBatch
from test.unit.udf_wrapper_params.question_answering.error_on_prediction_single_model_multiple_batch import \
    ErrorOnPredictionSingleModelMultipleBatch
from test.unit.udf_wrapper_params.question_answering.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from test.unit.udf_wrapper_params.question_answering.multiple_topk_multiple_size_single_model_multiple_batch_complete import \
    MultipleTopkMultipleSizeSingleModelNameMultipleBatch
from test.unit.udf_wrapper_params.question_answering.multiple_topk_multiple_size_single_model_single_batch_complete import \
    MultipleTopkMultipleSizeSingleModelNameSingleBatch
from test.unit.udf_wrapper_params.question_answering.multiple_topk_single_size_single_model_multiple_batch_complete import \
    MultipleTopkSingleSizeSingleModelNameMultipleBatch
from test.unit.udf_wrapper_params.question_answering.multiple_topk_single_size_single_model_single_batch_complete import \
    MultipleTopkSingleSizeSingleModelNameSingleBatch
from test.unit.udfs.output_matcher import Output, OutputMatcher
from test.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)


def create_mock_metadata(udf_wrapper):
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("question", str, "VARCHAR(2000000)"),
            Column("context_text", str, "VARCHAR(2000000)"),
            Column("top_k", int, "INTEGER")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("question", str, "VARCHAR(2000000)"),
            Column("context_text", str, "VARCHAR(2000000)"),
            Column("top_k", int, "INTEGER"),
            Column("answer", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("rank", int, "INTEGER"),
            Column("error_message", str, "VARCHAR(2000000)")
        ],
    )
    return meta


@pytest.mark.parametrize("params", [
    MultipleModelMultipleBatchComplete,
    MultipleTopkSingleSizeSingleModelNameSingleBatch,
    MultipleTopkSingleSizeSingleModelNameMultipleBatch,
    MultipleTopkMultipleSizeSingleModelNameSingleBatch,
    MultipleTopkMultipleSizeSingleModelNameMultipleBatch,
    ErrorOnPredictionSingleModelMultipleBatch,
    ErrorOnPredictionMultipleModelMultipleBatch
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_question_answering(mock_local_path, mock_create_loc, params):

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
            OutputMatcher(result_output, n_input_columns) == expected_output and
            params.mock_pipeline.counter == params.expected_model_counter)
    finally:
        params.mock_pipeline.counter = 0
