from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
    ErrorMessageMatcher,
    NewColumnsEmptyMatcher,
    NoErrorMessageMatcher,
    RankDTypeMatcher,
    RankMonotonicMatcher,
    Result,
    ScoreMatcher,
    ShapeMatcher,
)
from test.integration_tests.without_db.udfs.utils.mock_context import MockContext
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import MockExaEnvironment
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest
import torch

from exasol_transformers_extension.udfs.models.zero_shot_text_classification_udf import (
    ZeroShotTextClassificationUDF,
)


def prepare_bucketfs(
    prepare_zero_shot_classification_model_for_local_bucketfs
):
    bucketfs_base_path = prepare_zero_shot_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)
    return bucketfs_conn_name, bucketfs_connection


def base_params():
    n_rows = 3
    batch_size = 2
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    n_labels = len(candidate_labels.split(","))
    return n_rows, batch_size, candidate_labels, n_labels


def run_test(sample_data, columns, bucketfs_conn_name, bucketfs_connection, batch_size):
    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = ZeroShotTextClassificationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]

    return result_df


def format_result(result_df):
    grouped_by_inputs = result_df.groupby("text_data")
    n_unique_labels_per_input = grouped_by_inputs["label"].nunique().to_list()
    result = Result(result_df)
    return result, n_unique_labels_per_input

@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
@pytest.mark.parametrize(
    "return_ranks, number_results_per_input",
    [
        ("ALL", None),
        ("HIGHEST", 1)# todo does this make it download the model multiple times?
    ],
)
def test_zero_shot_classification_single_text_udf(
    description, device_id, return_ranks, number_results_per_input,
    prepare_zero_shot_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_conn_name, bucketfs_connection = prepare_bucketfs(
        prepare_zero_shot_classification_model_for_local_bucketfs
    )

    n_rows, batch_size, candidate_labels, n_labels = base_params()
    if not number_results_per_input:
        number_results_per_input = n_labels

    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.zero_shot_model_specs.model_name,
            model_params.text_data + str(i),
            candidate_labels + str(i),
            return_ranks,
        )
        for i in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "candidate_labels",
        "return_ranks",
    ]

    result_df = run_test(
        sample_data, columns, bucketfs_conn_name, bucketfs_connection, batch_size
    )
    result, n_unique_labels_per_input = format_result(result_df)
    new_columns = ["label", "score", "rank", "error_message"]

    # assertions
    result = Result(result_df)
    assert result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=number_results_per_input)
    assert (
        result == ScoreMatcher()
        and result == RankDTypeMatcher()
        and result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=number_results_per_input)
        and result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=n_rows,
            results_per_row=number_results_per_input,
        )
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == [number_results_per_input] * n_rows
    )


@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
@pytest.mark.parametrize(
    "return_ranks, number_results_per_input",
    [
        ("ALL", None),
        ("HIGHEST", 1)
    ],
)
def test_zero_shot_classification_single_text_udf_with_span(
    description, device_id, return_ranks, number_results_per_input,
        prepare_zero_shot_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_conn_name, bucketfs_connection = prepare_bucketfs(
        prepare_zero_shot_classification_model_for_local_bucketfs
    )

    n_rows, batch_size, candidate_labels, n_labels = base_params()
    if not number_results_per_input:
        number_results_per_input = n_labels

    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.zero_shot_model_specs.model_name,
            model_params.text_data + str(i),
            # model_params.text_data * (i + 1),
            i,
            0,
            len(model_params.text_data),
            candidate_labels + str(i),
            return_ranks,
        )
        for i in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "text_data_doc_id",
        "text_data_char_begin",
        "text_data_char_end",
        "candidate_labels",
        "return_ranks",
    ]

    result_df = run_test(
        sample_data, columns, bucketfs_conn_name, bucketfs_connection, batch_size
    )
    result, n_unique_labels_per_input = format_result(result_df)

    new_columns = ["label", "score", "rank", "error_message"]

    # assertions
    assert (
        result == ScoreMatcher()
        and result == RankDTypeMatcher()
        and result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=number_results_per_input)
        and result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=n_rows,
            results_per_row=number_results_per_input,
        )
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == [number_results_per_input] * n_rows
    )


@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
@pytest.mark.parametrize(
    "return_ranks, number_results_per_input",
    [
        ("ALL", 1),
        ("HIGHEST", 1)
    ],
)
def test_zero_shot_classification_single_text_udf_on_error_handling(
    description, device_id, return_ranks, number_results_per_input,
        prepare_zero_shot_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )
    bucketfs_conn_name, bucketfs_connection = (
        prepare_bucketfs(prepare_zero_shot_classification_model_for_local_bucketfs))

    n_rows, batch_size, candidate_labels, _ = base_params()
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            "not existing model",
            model_params.text_data + str(i),
            candidate_labels + str(i),
            return_ranks,
        )
        for i in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "text_data",
        "candidate_labels",
        "return_ranks",
    ]

    result_df = run_test(
        sample_data, columns, bucketfs_conn_name, bucketfs_connection, batch_size
    )
    new_columns = ["label", "score", "rank", "error_message"]

    result = Result(result_df)
    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows * number_results_per_input)
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NewColumnsEmptyMatcher(new_columns=new_columns)
        and result == ErrorMessageMatcher()
    )
