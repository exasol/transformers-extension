from test.integration_tests.without_db.udfs.utils.matcher import (
    ColumnsMatcher,
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

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_BUCKETFS_CONN_NAME,
)
from exasol_transformers_extension.udfs.models.ai_answer import AiAnswerUDF


@pytest.mark.parametrize(
    "description, n_rows",
    [
        ("on CPU with batch input, single answer", 3),
        ("on CPU with single input, single answer", 1),
    ],
)
def test_ai_answer_udf(
    description,
    n_rows,
    prepare_default_question_answering_model_for_local_bucketfs,
):
    bucketfs_base_path = prepare_default_question_answering_model_for_local_bucketfs
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    question = "Where is the company Exasol?"
    sample_data = [
        (
            question,
            model_params.text_data,
        )
        for _ in range(n_rows)
    ]
    columns = [
        "question",
        "context_text",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({DEFAULT_BUCKETFS_CONN_NAME: bucketfs_connection})

    sequence_classifier = AiAnswerUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_dfs = ctx.get_emitted()
    result_df = pd.concat(result_dfs)

    new_columns = ["answer", "error_message"]

    result = Result(result_df)

    assert result == ShapeMatcher(
        columns=columns, new_columns=new_columns, removed_columns=[], n_rows=n_rows
    )
    assert result == ColumnsMatcher(columns=columns, new_columns=new_columns)
    assert result == NoErrorMessageMatcher()
