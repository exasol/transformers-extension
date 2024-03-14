import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from test.unit.udf_wrapper_params.text_generation.\
    error_not_cached_multiple_model_multiple_batch import (
        ErrorNotCachedMultipleModelMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    error_not_cached_single_model_multiple_batch import (
        ErrorNotCachedSingleModelMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    error_on_prediction_multiple_model_multiple_batch import (
        ErrorOnPredictionMultipleModelMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    error_on_prediction_single_model_multiple_batch import (
        ErrorOnPredictionSingleModelMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_bfsconn_single_subdir_single_model_multiple_batch import (
        MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
        multiple_bfsconn_single_subdir_single_model_single_batch import (
        MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_max_length_single_model_multiple_batch import (
        MultipleMaxLengthSingleModelNameMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_max_length_single_model_single_batch import (
        MultipleMaxLengthSingleModelNameSingleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_model_multiple_batch_complete import (
        MultipleModelMultipleBatchComplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_model_multiple_batch_incomplete import (
        MultipleModelMultipleBatchIncomplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_model_multiple_batch_multiple_models_per_batch import (
        MultipleModelMultipleBatchMultipleModelsPerBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_model_single_batch_complete import (
        MultipleModelSingleBatchComplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_model_single_batch_incomplete import (
        MultipleModelSingleBatchIncomplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_return_full_param_single_model_multiple_batch import (
        MultipleReturnFullParamSingleModelNameMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    multiple_return_full_param_single_model_single_batch import (
        MultipleReturnFullParamSingleModelNameSingleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    single_bfsconn_multiple_subdir_single_model_multiple_batch import (
        SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    single_bfsconn_multiple_subdir_single_model_single_batch import (
        SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    )
from test.unit.udf_wrapper_params.text_generation.\
    single_model_multiple_batch_complete import (
        SingleModelMultipleBatchComplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    single_model_multiple_batch_incomplete import (
        SingleModelMultipleBatchIncomplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    single_model_single_batch_complete import (
        SingleModelSingleBatchComplete,
    )
from test.unit.udf_wrapper_params.text_generation.\
    single_model_single_batch_incomplete import (
        SingleModelSingleBatchIncomplete,
    )
from test.unit.udfs.output_matcher import (
    Output,
    OutputMatcher,
)


def create_mock_metadata(udf_wrapper):
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("max_length", int, "INTEGER"),
            Column("return_full_text", bool, "BOOLEAN"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("max_length", int, "INTEGER"),
            Column("return_full_text", bool, "BOOLEAN"),
            Column("generated_text", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ],
    )
    return meta


@pytest.mark.parametrize(
    "params",
    [
        SingleModelSingleBatchComplete,
        SingleModelSingleBatchIncomplete,
        SingleModelMultipleBatchComplete,
        SingleModelMultipleBatchIncomplete,
        MultipleModelSingleBatchComplete,
        MultipleModelSingleBatchIncomplete,
        MultipleModelMultipleBatchComplete,
        MultipleModelMultipleBatchIncomplete,
        MultipleModelMultipleBatchMultipleModelsPerBatch,
        SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
        SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
        MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
        MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
        MultipleMaxLengthSingleModelNameMultipleBatch,
        MultipleMaxLengthSingleModelNameSingleBatch,
        MultipleReturnFullParamSingleModelNameSingleBatch,
        MultipleReturnFullParamSingleModelNameMultipleBatch,
        ErrorNotCachedSingleModelMultipleBatch,
        ErrorNotCachedMultipleModelMultipleBatch,
        ErrorOnPredictionSingleModelMultipleBatch,
        ErrorOnPredictionMultipleModelMultipleBatch,
    ],
)
def test_text_generation(params):
    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)

    exa = MockExaEnvironment(metadata=meta, connections=params.bfs_connections)

    result = executor.run([Group(params.input_data)], exa)
    result_output = Output(result[0].rows)
    expected_output = Output(params.output_data)
    n_input_columns = len(meta.input_columns) - 1

    try:
        assert (
            OutputMatcher(result_output, n_input_columns) == expected_output,
            params.mock_pipeline.counter == params.expected_model_counter,
        )
    finally:
        params.mock_pipeline.counter = 0
