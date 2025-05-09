import re
from test.unit.utils.utils_for_base_udf_tests import (
    create_mock_metadata,
    regex_matcher,
    run_test,
)
from test.unit.utils.utils_for_udf_tests import (
    create_base_mock_model_factories,
    create_mock_exa_environment_with_token_con,
    create_mock_pipeline_factory,
    create_mock_udf_context,
)
from test.utils.mock_bucketfs_location import (
    fake_bucketfs_location_from_conn_object,
    fake_local_bucketfs_path,
)
from unittest.mock import patch

import pytest
from exasol_udf_mock_python.connection import Connection


def data_for_model_loader_tests(model_name: str, sub_dir: str, bucketfs_conn_name: str):
    input_data = [(1, model_name, sub_dir, bucketfs_conn_name, "")]
    output_data = [{"answer": "answer", "score": 1.0}]
    return input_data, output_data


def setup_model_loader_tests_and_run(
    bucketfs_conn_name, bucketfs_conn, input_data, model_output_data
):
    mock_base_model_factory, mock_tokenizer_factory = create_base_mock_model_factories()
    mock_meta = create_mock_metadata()
    mock_exa = create_mock_exa_environment_with_token_con(
        [bucketfs_conn_name], [bucketfs_conn], mock_meta, "", None
    )  # todo do we need empty token con

    mock_pipeline_factory = create_mock_pipeline_factory([[[model_output_data]]], 1)
    mock_ctx = create_mock_udf_context(input_data, mock_meta)
    res = run_test(
        mock_exa,
        mock_base_model_factory,
        mock_tokenizer_factory,
        mock_pipeline_factory,
        mock_ctx,
    )
    return res, mock_meta


@pytest.mark.parametrize(
    ["description", "bucketfs_conn_name", "bucketfs_conn", "sub_dir", "model_name"],
    [
        (
            "all given",
            "test_bucketfs_con_name",
            Connection(address="file:///test"),
            "test_subdir",
            "test_model",
        )
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_model_loader_all_parameters(
    mock_local_path,
    mock_create_loc,
    description,
    bucketfs_conn_name,
    bucketfs_conn,
    sub_dir,
    model_name,
):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data, model_output_data = data_for_model_loader_tests(
        model_name, sub_dir, bucketfs_conn_name
    )

    res, mock_meta = setup_model_loader_tests_and_run(
        bucketfs_conn_name, bucketfs_conn, input_data, model_output_data
    )
    # check if no errors
    assert res[0][-1] is None and len(res[0]) == len(mock_meta.output_columns)


@pytest.mark.parametrize(
    ["description", "bucketfs_conn_name", "bucketfs_conn", "sub_dir", "model_name"],
    [
        ("all null", None, None, None, None),
        (
            "model name missing",
            "test_bucketfs_con_name",
            Connection(address="file:///test"),
            "test_subdir",
            None,
        ),
        ("bucketfs_conn missing", None, None, "test_subdir", "test_model"),
        (
            "sub_dir missing",
            "test_bucketfs_con_name",
            Connection(address="file:///test"),
            None,
            "test_model",
        ),
        (
            "model_name missing",
            "test_bucketfs_con_name",
            Connection(address="file:///test"),
            "test_subdir",
            None,
        ),
    ],
)
@patch(
    "exasol.python_extension_common.connections.bucketfs_location.create_bucketfs_location_from_conn_object"
)
@patch(
    "exasol_transformers_extension.utils.bucketfs_operations.get_local_bucketfs_path"
)
def test_model_loader_missing_parameters(
    mock_local_path,
    mock_create_loc,
    description,
    bucketfs_conn_name,
    bucketfs_conn,
    sub_dir,
    model_name,
):

    mock_create_loc.side_effect = fake_bucketfs_location_from_conn_object
    mock_local_path.side_effect = fake_local_bucketfs_path

    input_data, model_output_data = data_for_model_loader_tests(
        model_name, sub_dir, bucketfs_conn_name
    )

    res, mock_meta = setup_model_loader_tests_and_run(
        bucketfs_conn_name, bucketfs_conn, input_data, model_output_data
    )

    error_field = res[0][-1]
    expected_error = regex_matcher(
        ".*For each model model_name, bucketfs_conn and sub_dir need to be provided."
        f" Found model_name = {model_name}, bucketfs_conn = .*, sub_dir = {sub_dir}.",
        flags=re.DOTALL,
    )
    assert error_field == expected_error
    assert error_field is not None and len(res[0]) == len(mock_meta.output_columns)
