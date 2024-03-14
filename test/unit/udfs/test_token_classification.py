import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from test.unit.udf_wrapper_params.token_classification.error_not_cached_multiple_model_multiple_batch import (
    ErrorNotCachedMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.error_not_cached_single_model_multiple_batch import (
    ErrorNotCachedSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.error_on_prediction_multiple_model_multiple_batch import (
    ErrorOnPredictionMultipleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.error_on_prediction_single_model_multiple_batch import (
    ErrorOnPredictionSingleModelMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.multiple_bfsconn_single_subdir_single_model_multiple_batch import (
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.multiple_bfsconn_single_subdir_single_model_single_batch import (
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
)
from test.unit.udf_wrapper_params.token_classification.multiple_model_multiple_batch_complete import (
    MultipleModelMultipleBatchComplete,
)
from test.unit.udf_wrapper_params.token_classification.multiple_model_multiple_batch_incomplete import (
    MultipleModelMultipleBatchIncomplete,
)
from test.unit.udf_wrapper_params.token_classification.multiple_model_multiple_batch_multiple_models_per_batch import (
    MultipleModelMultipleBatchMultipleModelsPerBatch,
)
from test.unit.udf_wrapper_params.token_classification.multiple_model_single_batch_complete import (
    MultipleModelSingleBatchComplete,
)
from test.unit.udf_wrapper_params.token_classification.multiple_model_single_batch_incomplete import (
    MultipleModelSingleBatchIncomplete,
)
from test.unit.udf_wrapper_params.token_classification.multiple_strategy_single_model_multiple_batch import (
    MultipleStrategySingleModelNameMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.multiple_strategy_single_model_single_batch import (
    MultipleStrategySingleModelNameSingleBatch,
)
from test.unit.udf_wrapper_params.token_classification.single_bfsconn_multiple_subdir_single_model_multiple_batch import (
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
)
from test.unit.udf_wrapper_params.token_classification.single_bfsconn_multiple_subdir_single_model_single_batch import (
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
)
from test.unit.udf_wrapper_params.token_classification.single_model_multiple_batch_complete import (
    SingleModelMultipleBatchComplete,
)
from test.unit.udf_wrapper_params.token_classification.single_model_multiple_batch_incomplete import (
    SingleModelMultipleBatchIncomplete,
)
from test.unit.udf_wrapper_params.token_classification.single_model_single_batch_complete import (
    SingleModelSingleBatchComplete,
)
from test.unit.udf_wrapper_params.token_classification.single_model_single_batch_incomplete import (
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
            Column("aggregation_strategy", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("token_conn", str, "VARCHAR(2000000)"),
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
        SingleModelSingleBatchIncomplete,
        SingleModelSingleBatchComplete,
        SingleModelMultipleBatchIncomplete,
        SingleModelMultipleBatchComplete,
        MultipleModelSingleBatchIncomplete,
        MultipleModelSingleBatchComplete,
        MultipleModelMultipleBatchIncomplete,
        MultipleModelMultipleBatchComplete,
        MultipleModelMultipleBatchMultipleModelsPerBatch,
        SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
        SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
        MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
        MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
        MultipleStrategySingleModelNameSingleBatch,
        MultipleStrategySingleModelNameMultipleBatch,
        ErrorNotCachedSingleModelMultipleBatch,
        ErrorNotCachedMultipleModelMultipleBatch,
        ErrorOnPredictionMultipleModelMultipleBatch,
        ErrorOnPredictionSingleModelMultipleBatch,
    ],
)
def test_token_classification(params):
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
