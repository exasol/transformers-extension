import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from tests.unit_tests.udf_wrapper_params.text_generation.multiple_bfsconn_single_subdir_single_model_multiple_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_bfsconn_single_subdir_single_model_single_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_max_length_single_model_multiple_batch import \
    MultipleMaxLengthSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_max_length_single_model_single_batch import \
    MultipleMaxLengthSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_model_multiple_batch_incomplete import \
    MultipleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_model_multiple_batch_multiple_models_per_batch import \
    MultipleModelMultipleBatchMultipleModelsPerBatch
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_model_single_batch_complete import \
    MultipleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_model_single_batch_incomplete import \
    MultipleModelSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_return_full_param_single_model_multiple_batch import \
    MultipleReturnFullParamSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.multiple_return_full_param_single_model_single_batch import \
    MultipleReturnFullParamSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.single_bfsconn_multiple_subdir_single_model_multiple_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.single_bfsconn_multiple_subdir_single_model_single_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.text_generation.single_model_multiple_batch_complete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.text_generation.single_model_multiple_batch_incomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.text_generation.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.text_generation.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete


def create_mock_metadata(udf_wrapper):
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("max_length", int, "INTEGER"),
            Column("return_full_text", bool, "BOOLEAN")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("max_length", int, "INTEGER"),
            Column("return_full_text", bool, "BOOLEAN"),
            Column("generated_text", str, "VARCHAR(2000000)")
        ],
    )
    return meta


@pytest.mark.parametrize("params", [
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
    MultipleReturnFullParamSingleModelNameMultipleBatch
])
def test_text_generation(params):
    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)

    exa = MockExaEnvironment(
        metadata=meta,
        connections=params.bfs_connections)

    result = executor.run([Group(params.input_data)], exa)
    assert result[0].rows == params.output_data
