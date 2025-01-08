from typing import Union, List
from unittest.mock import patch, MagicMock, create_autospec

import pytest
from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_context import StandaloneMockContext
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from transformers import AutoModel, Pipeline

from exasol_transformers_extension.udfs.models.zero_shot_text_classification_udf import ZeroShotTextClassificationUDF
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from tests.unit_tests.udf_wrapper_params.zero_shot.error_on_prediction_multiple_model_multiple_batch import \
    ErrorOnPredictionMultipleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.zero_shot.error_on_prediction_single_model_multiple_batch import \
    ErrorOnPredictionSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.zero_shot.multiple_labels_single_model_multiple_batch import \
    MultipleLabelsSingleModelMultipleBatch
from tests.unit_tests.udf_wrapper_params.zero_shot.multiple_labels_single_model_single_batch import \
    MultipleLabelsSingleModelSingleBatch
from tests.unit_tests.udf_wrapper_params.zero_shot.multiple_model_multiple_batch_complete import \
    MultipleModelMultipleBatchComplete
from tests.unit_tests.utils.utils_for_udf_tests import assert_correct_number_of_results, \
    assert_result_matches_expected_output

from tests.utils.mock_bucketfs_location import (fake_bucketfs_location_from_conn_object, fake_local_bucketfs_path)
from tests.utils.mock_cast import mock_cast


def create_mock_metadata():
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("candidate_labels", str, "VARCHAR(2000000)")
        ],
        output_type="EMITS",
        output_columns=[
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("text_data", str, "VARCHAR(2000000)"),
            Column("candidate_labels", str, "VARCHAR(2000000)"),
            Column("label", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("rank", int, "INTEGER"),
            Column("error_message", str, "VARCHAR(2000000)")
        ],
    )
    return meta

#todo move these to utils instead of copy them?
def create_db_mocks(bfs_connection, model_input_data, mock_meta):
    mock_ctx = StandaloneMockContext(inp=model_input_data, metadata=mock_meta)
    mock_exa = MockExaEnvironment(
        metadata=mock_meta,
        connections=bfs_connection)
    return mock_ctx, mock_exa

def create_mock_model_factorys(number_of_intended_used_models):
    """
    Creates mocks for transformers.AutoModel and gives them to mocks a base_model_factory_mock as side_effect.
    This way mock_base_model_factory can the return a mock_model when called by the udf.
    In test cases where we expect the model loading to fail, we create only expected model, and then try loading
    more which results in no model being returned triggering our exception.
    mock_tokenizer_factory does not need to return anything for our tests.
    """
    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol,
                                                                                      _name="mock_base_model_factory")
    mock_models: List[Union[AutoModel, MagicMock]] = [
        create_autospec(AutoModel) for i in range (0,number_of_intended_used_models)
        ]
    mock_cast(mock_base_model_factory.from_pretrained).side_effect = mock_models

    return mock_base_model_factory, mock_tokenizer_factory

def create_mock_pipeline_factory(tokenizer_models_output_df, number_of_intended_used_models):
    """
    Creates a mock pipeline (Normally created form model and tokenizer, then called with the data and outputs results).
    Ths mock gets a list of tokenizer_models_outputs as side_effect, enabling it to return them in order when called.
    if the specific tokenizer_models_output is a non-valid result (outputs are None), we give a
    list containing an Exception instead, so the mock can throw to test error_on_prediction.
    This mock_pipeline is feed into a mock_pipeline_factory.
    """
    mock_pipeline: List[Union[AutoModel, MagicMock]] = [
        create_autospec(Pipeline, side_effect=tokenizer_models_output_df[i])
        for i in range(0, number_of_intended_used_models)
        ]

    mock_pipeline_factory: Union[Pipeline, MagicMock] = create_autospec(Pipeline,
                                                                        side_effect=mock_pipeline)
    return mock_pipeline_factory


@pytest.mark.parametrize("params", [
    MultipleModelMultipleBatchComplete,
    MultipleLabelsSingleModelSingleBatch,
    MultipleLabelsSingleModelMultipleBatch,
    ErrorOnPredictionMultipleModelMultipleBatch,
    ErrorOnPredictionSingleModelMultipleBatch
])
@patch('exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object')
@patch('exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path')
def test_zero_shot(mock_local_path, mock_create_loc, params):
    """
    This test checks combinations of input data to determine correct output data. For this everything the udf uses in
    the background is mocked, and given to the udf. we then check if the resulting output matches the expected output.
    """
    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    model_input_data = params.input_data
    bfs_connection = params.bfs_connections
    expected_model_counter = params.expected_model_counter
    zero_shot_models_output_df = params.zero_shot_models_output_df
    batch_size = params.batch_size
    expected_output_data = params.output_data

    mock_meta = create_mock_metadata()
    mock_ctx, mock_exa = create_db_mocks(bfs_connection, model_input_data, mock_meta)
    mock_base_model_factory, mock_tokenizer_factory = create_mock_model_factorys(expected_model_counter)
    mock_pipeline_factory = create_mock_pipeline_factory(zero_shot_models_output_df, expected_model_counter)


    udf = ZeroShotTextClassificationUDF(exa=mock_exa,
                                        batch_size=batch_size,
                                        base_model=mock_base_model_factory,
                                        tokenizer=mock_tokenizer_factory,
                                        pipeline=mock_pipeline_factory)


    udf.run(mock_ctx)
    result = mock_ctx.output

    assert_correct_number_of_results(result, mock_meta.output_columns, expected_output_data)
    assert_result_matches_expected_output(result, expected_output_data,  mock_meta.input_columns)
    assert len(mock_pipeline_factory.mock_calls) == expected_model_counter