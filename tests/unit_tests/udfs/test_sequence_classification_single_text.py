import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor
from tests.unit_tests.udf_wrapper_params.sequence_classification.MultipleModelMultipleBatchComplete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.MultipleModelMultipleBatchIncomplete import \
    MultipleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.MultipleModelMultipleBatchMultipleModelsPerBatch import \
    MultipleModelMultipleBatchMultipleModelsPerBatch
from tests.unit_tests.udf_wrapper_params.sequence_classification.MultipleModelSingleBatchComplete import \
    MultipleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.MultipleModelSingleBatchIncomplete import \
    MultipleModelSingleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.SingleModelMultipleBatchComplete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.SingleModelMultipleBatchIncomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.SingleModelSingleBatchComplete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.sequence_classification.SingleModelSingleBatchIncomplete import \
    SingleModelSingleBatchIncomplete

BFS_CONN_NAME = "test_bfs_conn_name"
LABEL_SCORE_MAP = {'label_1': 0.21, 'label_2': 0.24,
                   'label_3': 0.26, 'label_4': 0.29}


def create_mock_metadata(udf_wrapper):
    meta = MockMetaData(
        script_code_wrapper_function=udf_wrapper,
        input_type="SET",
        input_columns=[
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
    SingleModelSingleBatchComplete(),
    SingleModelSingleBatchIncomplete(),
    SingleModelMultipleBatchComplete(),
    SingleModelMultipleBatchIncomplete()
])
def test_sequence_classification_single_text_single_model(
        params, upload_dummy_model_to_local_bucketfs):
    model_metadata = upload_dummy_model_to_local_bucketfs[0]

    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)
    bucketfs_connection = Connection(address=f"file://{model_metadata[1]}")
    exa = MockExaEnvironment(
        metadata=meta,
        connections={BFS_CONN_NAME: bucketfs_connection})

    input_data = []
    for text in params.single_text:
        input_data.append((
            BFS_CONN_NAME, model_metadata[0], model_metadata[1], text))
    result = executor.run([Group(input_data)], exa)

    rounded_result = _get_rounded_result(result)
    expected_result = _prepare_expected_result(input_data)
    assert rounded_result == expected_result


@pytest.mark.parametrize("params", [
    MultipleModelMultipleBatchComplete
])
def test_sequence_classification_single_text_multiple_models_with_factory(
        params, upload_dummy_model_to_local_bucketfs):
    models_metadata = upload_dummy_model_to_local_bucketfs

    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)
    bucketfs_connection = Connection(address=f"file://{models_metadata[0][1]}")
    exa = MockExaEnvironment(
        metadata=meta,
        connections={BFS_CONN_NAME: bucketfs_connection})

    input_data = None
    expected_result = None
    for model_metadata in models_metadata:
        input_data = [(BFS_CONN_NAME, model_metadata[0], model, text)
                      for model, text in params.inputs]
        expected_result = [(BFS_CONN_NAME, model_metadata[0], model, text, label, score)
                                for model, text, label, score in params.outputs]

    result = executor.run([Group(input_data)], exa)

    for row in result[0].rows:
        print(row)
    rounded_actual_result = [row[:5] + (round(row[5], 2),)
                             for row in result[0].rows]
    assert rounded_actual_result == expected_result


@pytest.mark.parametrize("params", [
    MultipleModelSingleBatchComplete(),
    MultipleModelSingleBatchIncomplete(),
    MultipleModelMultipleBatchIncomplete(),
    MultipleModelMultipleBatchMultipleModelsPerBatch()
])
def test_sequence_classification_single_text_multiple_models(
        params, upload_dummy_model_to_local_bucketfs):
    models_metadata = upload_dummy_model_to_local_bucketfs

    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)
    bucketfs_connection = Connection(address=f"file://{models_metadata[0][1]}")
    exa = MockExaEnvironment(
        metadata=meta,
        connections={BFS_CONN_NAME: bucketfs_connection})

    input_data = []
    for model_metadata in models_metadata:
        for text in params.single_text:
            input_data.append((
                BFS_CONN_NAME, model_metadata[0], model_metadata[1], text))
    result = executor.run([Group(input_data)], exa)

    rounded_result = _get_rounded_result(result)
    expected_result = _prepare_expected_result(input_data)
    assert rounded_result == expected_result


def _get_rounded_result(result):
    rounded_result = result[0].rows
    for i in range(len(rounded_result)):
        rounded_result[i] = rounded_result[i][:-1] + \
                            (round(rounded_result[i][-1], 2),)
    return rounded_result


def _prepare_expected_result(input_data):
    expected_result = []
    for i in range(len(input_data)):
        for label, score in sorted(LABEL_SCORE_MAP.items()):
            expected_result.append((input_data[i] + (label, score)))
    return expected_result
