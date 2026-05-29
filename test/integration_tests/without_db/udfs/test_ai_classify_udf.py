from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
    NoErrorMessageMatcher,
    RankDTypeMatcher,
    Result,
    ScoreMatcher,
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

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)
from exasol_transformers_extension.udfs.models.ai_classify_udf import AiClassifyUDF


def test_ai_classify_udf(
    prepare_default_classify_model_for_local_bucketfs,
):

    bucketfs_base_path = prepare_default_classify_model_for_local_bucketfs
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    n_rows = 3
    batch_size = 2
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    sample_data = [
        (model_params.text_data + str(i), candidate_labels) for i in range(n_rows)
    ]
    columns = [
        "text_data",
        "candidate_labels",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({DEFAULT_BUCKETFS_CONN_NAME: bucketfs_connection})

    sequence_classifier = AiClassifyUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_dfs = ctx.get_emitted()
    result_df = pd.concat(result_dfs)
    new_columns = ["label", "score", "error_message"]

    grouped_by_inputs = result_df.groupby("text_data")
    n_unique_labels_per_input = grouped_by_inputs["label"].nunique().to_list()
    result = Result(result_df)
    assert (
        result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            removed_columns=[],
            n_rows=n_rows,
            results_per_row=1,
        )
        and result == ColumnsMatcher(columns=columns, new_columns=new_columns)
        and result == ScoreMatcher()
        and result == NoErrorMessageMatcher()
        and n_unique_labels_per_input == [1] * n_rows
    )
