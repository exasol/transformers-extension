import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from tests.unit_tests.udf_wrapper_params.question_answering.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_model_multiple_batch_incomplete import \
    MultipleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_model_multiple_batch_multiple_models_per_batch import \
    MultipleModelMultipleBatchMultipleModelsPerBatch
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_model_single_batch_complete import \
    MultipleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_model_single_batch_incomplete import \
    MultipleModelSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_multiple_batch_complete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_multiple_batch_incomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete

BFS_CONN_NAME = "test_bfs_conn_name"


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
            Column("context_text", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("question", str, "VARCHAR(2000000)"),
            Column("context_text", str, "VARCHAR(2000000)"),
            Column("answer", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE")
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
    MultipleModelMultipleBatchMultipleModelsPerBatch
])
def test_question_answering(params, get_local_bucketfs_path):
    bucketfs_base_path = get_local_bucketfs_path

    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")
    exa = MockExaEnvironment(
        metadata=meta,
        connections={BFS_CONN_NAME: bucketfs_connection})

    input_data = [(input[0], BFS_CONN_NAME) + input[1:]
                  for input in params.input_data]
    result = executor.run([Group(input_data)], exa)

    expected_result = [(BFS_CONN_NAME, ) + output
                       for output in params.output_data]
    assert result[0].rows == expected_result

