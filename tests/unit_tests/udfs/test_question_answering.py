import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.group import Group
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from exasol_udf_mock_python.udf_mock_executor import UDFMockExecutor

from tests.unit_tests.udf_wrapper_params.question_answering.error_not_cached_multiple_model_multiple_batch import \
    ErrorNotCachedMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.error_not_cached_single_model_multiple_batch import \
    ErrorNotCachedSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.error_on_prediction_multiple_model_multiple_batch import \
    ErrorOnPredictionMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.error_on_prediction_single_model_multiple_batch import \
    ErrorOnPredictionSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_bfsconn_single_subdir_single_model_multiple_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_bfsconn_single_subdir_single_model_single_batch import \
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch
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
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_topk_multiple_size_single_model_multiple_batch_complete import \
    MultipleTopkMultipleSizeSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_topk_multiple_size_single_model_single_batch_complete import \
    MultipleTopkMultipleSizeSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_topk_single_size_single_model_multiple_batch_complete import \
    MultipleTopkSingleSizeSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.multiple_topk_single_size_single_model_single_batch_complete import \
    MultipleTopkSingleSizeSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.single_bfsconn_multiple_subdir_single_model_multiple_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.single_bfsconn_multiple_subdir_single_model_single_batch import \
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_multiple_batch_complete import \
    SingleModelMultipleBatchComplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_multiple_batch_incomplete import \
    SingleModelMultipleBatchIncomplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_single_batch_complete import \
    SingleModelSingleBatchComplete
from tests.unit_tests.udf_wrapper_params.question_answering.single_model_single_batch_incomplete import \
    SingleModelSingleBatchIncomplete
from tests.unit_tests.udfs.output_matcher import Output, OutputMatcher


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
    SingleModelSingleBatchIncomplete,
    SingleModelSingleBatchComplete,
    SingleModelMultipleBatchComplete,
    SingleModelMultipleBatchIncomplete,
    MultipleModelSingleBatchComplete,
    MultipleModelSingleBatchIncomplete,
    MultipleModelMultipleBatchComplete,
    MultipleModelMultipleBatchIncomplete,
    MultipleModelMultipleBatchMultipleModelsPerBatch,
    MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch,
    MultipleBucketFSConnSingleSubdirSingleModelNameMultipleBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch,
    SingleBucketFSConnMultipleSubdirSingleModelNameMultipleBatch,
    MultipleTopkSingleSizeSingleModelNameSingleBatch,
    MultipleTopkSingleSizeSingleModelNameMultipleBatch,
    MultipleTopkMultipleSizeSingleModelNameSingleBatch,
    MultipleTopkMultipleSizeSingleModelNameMultipleBatch,
    ErrorNotCachedSingleModelMultipleBatch,
    ErrorNotCachedMultipleModelMultipleBatch,
    ErrorOnPredictionSingleModelMultipleBatch,
    ErrorOnPredictionMultipleModelMultipleBatch
])
def test_question_answering(params):
    executor = UDFMockExecutor()
    meta = create_mock_metadata(params.udf_wrapper)

    exa = MockExaEnvironment(
        metadata=meta,
        connections=params.bfs_connections)

    result = executor.run([Group(params.input_data)], exa)
    result_output = Output(result[0].rows)
    expected_output = Output(params.output_data)

    n_input_columns = len(meta.input_columns) - 1
    assert OutputMatcher(result_output, n_input_columns) == expected_output
