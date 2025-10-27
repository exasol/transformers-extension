from test.unit.udfs.output_matcher import (
    Output,
    OutputMatcher,
)
from test.utils.mock_cast import mock_cast
from typing import (
    Any,
    List,
    Tuple,
    Union,
)
from unittest.mock import (
    MagicMock,
    create_autospec,
)

from exasol_udf_mock_python.connection import Connection
from exasol_udf_mock_python.mock_context import StandaloneMockContext
from exasol_udf_mock_python.mock_exa_environment import MockExaEnvironment
from exasol_udf_mock_python.mock_meta_data import MockMetaData
from transformers import (
    AutoModel,
    Pipeline,
)

from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)


def create_mock_udf_context(
    input_data: list[tuple[Any, ...]], mock_meta: MockMetaData
) -> StandaloneMockContext:
    mock_ctx = StandaloneMockContext(
        inp=input_data,
        metadata=mock_meta,
    )
    return mock_ctx


def create_mock_exa_environment(
    mock_meta: MockMetaData, bfs_connections: dict[str, Connection]
) -> MockExaEnvironment:
    mock_exa = MockExaEnvironment(metadata=mock_meta, connections=bfs_connections)
    return mock_exa


def create_bfs_connections_with_token_con(
    bfs_conn_names: list[str],
    bucketfs_connections: list[Connection],
    token_conn_name: str,
    token_conn_obj: Connection,
) -> dict[str, Connection]:
    connections_dict = dict(zip(bfs_conn_names, bucketfs_connections))
    connections_dict[token_conn_name] = token_conn_obj
    return connections_dict


def create_mock_exa_environment_with_token_con(
    bfs_conn_names: list[str],
    bucketfs_connections: list[Connection],
    mock_meta: MockMetaData,
    token_conn_name: str,
    token_conn_obj: Connection,
) -> MockExaEnvironment:
    bfs_connections = create_bfs_connections_with_token_con(
        bfs_conn_names, bucketfs_connections, token_conn_name, token_conn_obj
    )
    mock_exa = create_mock_exa_environment(mock_meta, bfs_connections)
    return mock_exa


def create_base_mock_model_factories():
    mock_tokenizer_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(
        ModelFactoryProtocol
    )
    mock_base_model_factory: Union[ModelFactoryProtocol, MagicMock] = create_autospec(
        ModelFactoryProtocol, _name="mock_base_model_factory"
    )
    return mock_tokenizer_factory, mock_base_model_factory


def create_mock_model_factories_with_models(number_of_intended_used_models: int):
    """
    Creates mocks for transformers.AutoModel and gives them to mocks a base_model_factory_mock as side_effect.
    This way mock_base_model_factory can the return a mock_model when called by the udf.
    In test cases where we expect the model loading to fail, we create only expected model, and then try loading
    more which results in no model being returned triggering our exception.
    mock_tokenizer_factory does not need to return anything for our test.
    """
    mock_tokenizer_factory, mock_base_model_factory = create_base_mock_model_factories()

    mock_models: list[Union[AutoModel, MagicMock]] = [
        create_autospec(AutoModel) for i in range(number_of_intended_used_models)
    ]
    mock_cast(mock_base_model_factory.from_pretrained).side_effect = mock_models

    return mock_base_model_factory, mock_tokenizer_factory


def create_mock_pipeline_factory(
    tokenizer_models_output_df, number_of_intended_used_models
):
    """
    Creates a mock pipeline (Normally created form model and tokenizer, then called with the data and outputs results).
    Ths mock gets a list of tokenizer_models_outputs as side_effect, enabling it to return them in order when called.
    This mock_pipeline is feed into a mock_pipeline_factory.
    """
    mock_pipeline: list[Union[AutoModel, MagicMock]] = [
        create_autospec(Pipeline, side_effect=tokenizer_models_output_df[i])
        for i in range(0, number_of_intended_used_models)
    ]

    mock_pipeline_factory: Union[Pipeline, MagicMock] = create_autospec(
        Pipeline, side_effect=mock_pipeline
    )
    return mock_pipeline_factory


def assert_correct_number_of_results(result, output_columns, output_data):
    assert len(result[0]) == len(output_columns), (
        f"Number of columns in result is {len(result[0])},"
        f"not as expected {len(output_columns)}"
    )
    assert len(result) == len(output_data), (
        f"Number of lines in result is {len(result)}, "
        f"not as expected {len(output_data)}"
    )


def assert_result_matches_expected_output(result, expected_output_data, input_columns):
    expected_output = Output(expected_output_data)
    actual_output = Output(result)
    n_input_columns = len(input_columns) - 1
    assert OutputMatcher(actual_output, n_input_columns) == expected_output, (
        "OutputMatcher found expected_output_data "
        "and result not matching:"
        "expected_output_data: \n"
        f"{expected_output_data}\n"
        "actual_output_data: \n"
        f"{actual_output}"
    )


def assert_result_matches_expected_output_order_agnostic(
    result, expected_output_data, input_columns, sort_by_column
):
    expected_output = Output(expected_output_data)
    actual_output = Output(result)
    n_input_columns = len(input_columns) - 1
    assert OutputMatcher(actual_output, n_input_columns).equal_order_agnostic(
        expected_output, sort_by_column
    ), (
        "OutputMatcher found expected_output_data "
        "and result not matching:"
        "expected_output_data: \n"
        f"{expected_output_data}\n"
        "actual_output_data: \n"
        f"{actual_output}"
    )


def make_number_of_strings(input_str: str, desired_number: int):
    """
    Returns desired number of "input_strX", where X is counting up to desired_number.
    """
    return (input_str + f"{i}" for i in range(desired_number))
