from test.integration_tests.without_db.udfs.matcher import (
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
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params
from typing import Dict

import pandas as pd
import pytest
import torch
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.zero_shot_text_classification_udf import (
    ZeroShotTextClassificationUDF,
)


class ExaEnvironment:
    def __init__(self, connections: Dict[str, Connection] = None):
        self._connections = connections
        if self._connections is None:
            self._connections = {}

    def get_connection(self, name: str) -> Connection:
        return self._connections[name]


class Context:
    def __init__(self, input_df):
        self.input_df = input_df
        self._emitted = []
        self._is_accessed_once = False

    def emit(self, *args):
        self._emitted.append(args)

    def reset(self):
        self._is_accessed_once = False

    def get_emitted(self):
        return self._emitted

    def get_dataframe(self, num_rows="all", start_col=0):
        return_df = (
            None
            if self._is_accessed_once
            else self.input_df[self.input_df.columns[start_col:]]
        )
        self._is_accessed_once = True
        return return_df


def prepare_bucketfs(
    prepare_zero_shot_classification_model_for_local_bucketfs,
    bfs_conn="bucketfs_connection",
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

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

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
def test_zero_shot_classification_single_text_udf(
    description, device_id, prepare_zero_shot_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_conn_name, bucketfs_connection = prepare_bucketfs(
        prepare_zero_shot_classification_model_for_local_bucketfs
    )

    n_rows, batch_size, candidate_labels, n_labels = base_params()
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.zero_shot_model_specs.model_name,
            model_params.text_data + str(i),
            candidate_labels + str(i),
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
        and result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=n_labels)
        and result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=n_rows,
            results_per_row=n_labels,
        )
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == [n_labels] * n_rows
    )


@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
def test_zero_shot_classification_single_text_udf_with_span(
    description, device_id, prepare_zero_shot_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_conn_name, bucketfs_connection = prepare_bucketfs(
        prepare_zero_shot_classification_model_for_local_bucketfs
    )

    n_rows, batch_size, candidate_labels, n_labels = base_params()
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
        and result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=n_labels)
        and result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=n_rows,
            results_per_row=n_labels,
        )
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == [n_labels] * n_rows
    )


@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
def test_zero_shot_classification_single_text_udf_on_error_handling(
    description, device_id, prepare_zero_shot_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_zero_shot_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            "not existing model",
            model_params.text_data + str(i),
            candidate_labels + str(i),
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
    ]
    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = ZeroShotTextClassificationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["label", "score", "rank", "error_message"]

    result = Result(result_df)
    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NewColumnsEmptyMatcher(new_columns=new_columns)
        and result == ErrorMessageMatcher()
    )
