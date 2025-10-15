from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
    ErrorMessageMatcher,
    NewColumnsEmptyMatcher,
    NoErrorMessageMatcher,
    Result,
    ShapeMatcher,
)
from test.integration_tests.without_db.udfs.utils.mock_context import MockContext
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import MockExaEnvironment
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest
import torch
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.sequence_classification_single_text_udf import (
    SequenceClassificationSingleTextUDF,
)



@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
def test_sequence_classification_single_text_udf(
    description, device_id, prepare_sequence_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_sequence_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    n_rows = 3
    batch_size = 2
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.sequence_class_model_specs.model_name,
            model_params.text_data + str(i),
        )
        for i in range(n_rows)
    ]
    columns = ["device_id", "bucketfs_conn", "sub_dir", "model_name", "text_data"]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationSingleTextUDF(
        exa, batch_size=batch_size
    )
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["label", "score", "error_message"]

    grouped_by_inputs = result_df.groupby("text_data")
    n_unique_labels_per_input = grouped_by_inputs["label"].nunique().to_list()
    n_labels = 3
    n_labels_per_input_expected = [n_labels] * n_rows
    result = Result(result_df)
    assert (
        result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=n_rows,
            results_per_row=n_labels,
        )
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == n_labels_per_input_expected
    )


@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
def test_sequence_classification_single_text_udf_on_error_handling(
    description, device_id, prepare_sequence_classification_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_sequence_classification_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            "not existing model",
            model_params.text_data + str(i),
        )
        for i in range(n_rows)
    ]
    columns = ["device_id", "bucketfs_conn", "sub_dir", "model_name", "text_data"]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationSingleTextUDF(
        exa, batch_size=batch_size
    )
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["label", "score", "error_message"]

    result = Result(result_df)
    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
        and result == NewColumnsEmptyMatcher(new_columns=new_columns)
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == ErrorMessageMatcher()
    )
