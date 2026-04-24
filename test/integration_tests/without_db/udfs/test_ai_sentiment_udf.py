from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_BUCKETFS_CONN_NAME
from exasol_transformers_extension.udfs.models.ai_sentiment_udf import AiSentimentUDF
from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
    ErrorMessageMatcher,
    NewColumnsEmptyMatcher,
    NoErrorMessageMatcher,
    Result,
    ShapeMatcher,
)
from test.integration_tests.without_db.udfs.utils.mock_context import MockContext
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import (
    MockExaEnvironment,
)
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("description, device_id", [("on CPU", None), ("on GPU", 0)])
def test_ai_sentiment_extended_udf(
    description, device_id, prepare_default_sentiment_model_for_local_bucketfs
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) to execute the test"
        )

    bucketfs_base_path = prepare_default_sentiment_model_for_local_bucketfs
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    n_rows = 3
    batch_size = 2
    sample_data = [
        (
            model_params.text_data + str(i),
        )
        for i in range(n_rows)
    ]
    columns = [
        "text_data",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({DEFAULT_BUCKETFS_CONN_NAME: bucketfs_connection})

    sequence_classifier = AiSentimentUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_dfs = ctx.get_emitted()
    result_df = pd.concat(result_dfs)

    new_columns = ["label", "score", "error_message"]

    grouped_by_inputs = result_df.groupby("text_data")
    n_unique_labels_per_input = grouped_by_inputs["label"].nunique().to_list()
    n_labels = 1
    n_labels_per_input_expected = [n_labels] * n_rows
    result = Result(result_df)
    assert (
        result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            removed_columns=[],
            n_rows=n_rows,
            results_per_row=n_labels,
        )
        and result == ColumnsMatcher(columns=columns, new_columns=new_columns)
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == n_labels_per_input_expected
    )

