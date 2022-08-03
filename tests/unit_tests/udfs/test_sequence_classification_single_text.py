import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_locations_multiple_batch_incomplete import \
    MultipleLocationsMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_locations_multiple_batch_multiple_locations_per_batch import \
    MultipleModelLocationsMultipleBatchMultipleLocationsPerBatch
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_locations_single_batch_complete import \
    MultipleLocationsSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_locations_single_batch_incomplete import \
    MultipleLocationsSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_model_multiple_batch_incomplete import \
    MultipleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_model_multiple_batch_multiple_models_per_batch import \
    MultipleModelMultipleBatchMultipleModelsPerBatch
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_model_single_batch_complete import \
    MultipleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.multiple_model_single_batch_incomplete import \
    MultipleModelSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.single_model_multiple_batch_complete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.single_model_multiple_batch_incomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete
from tests.utils import postprocessing


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
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("label", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
        ],
    )
    return meta


@pytest.mark.parametrize("params", [
    SingleModelSingleBatchComplete,
    SingleModelSingleBatchIncomplete,
    SingleModelMultipleBatchComplete,
    SingleModelMultipleBatchIncomplete,
    MultipleModelMultipleBatchComplete,
    MultipleModelMultipleBatchIncomplete,
    MultipleModelSingleBatchComplete,
    MultipleModelSingleBatchIncomplete,
    MultipleModelMultipleBatchMultipleModelsPerBatch,
    MultipleLocationsSingleBatchComplete,
    MultipleLocationsSingleBatchIncomplete,
    MultipleLocationsMultipleBatchIncomplete,
    MultipleModelLocationsMultipleBatchMultipleLocationsPerBatch
])
def test_sequence_classification_single_text(params, get_local_bucketfs_path):
    bucketfs_base_path = get_local_bucketfs_path

    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper_single_text)

    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")
    exa = MockExaEnvironment(
        metadata=meta,
        connections={
            "bfs_conn1": bucketfs_connection,
            "bfs_conn2": bucketfs_connection,
            "bfs_conn3": bucketfs_connection,
            "bfs_conn4": bucketfs_connection})

    result = executor.run([Group(params.inputs_single_text)], exa)
    rounded_actual_result = postprocessing.get_rounded_result(result)
    assert rounded_actual_result == params.outputs_single_text
