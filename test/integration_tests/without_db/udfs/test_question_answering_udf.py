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
from test.integration_tests.without_db.udfs.utils.mock_exa_environment import (
    MockExaEnvironment,
)
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

import pandas as pd
import pytest
import torch
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.question_answering_udf import (
    QuestionAnsweringUDF,
)


@pytest.mark.parametrize(
    "description,  device_id, n_rows, top_k",
    [
        ("on CPU with batch input, single answer", None, 3, 1),
        ("on CPU with batch input, multiple answers", None, 3, 2),
        ("on CPU with single input, single answer", None, 1, 1),
        ("on CPU with single input, multiple answers", None, 1, 2),
        ("on GPU with batch input, single answer", 0, 3, 1),
        ("on GPU with batch input, multiple answers", 0, 3, 2),
        ("on GPU with single input, single answer", 0, 1, 1),
        ("on GPU with single input, multiple answers", 0, 1, 2),
    ],
)
def test_question_answering_udf(
    description,
    device_id,
    n_rows,
    top_k,
    prepare_question_answering_model_for_local_bucketfs,
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_question_answering_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    question = "Where is the Exasol?"
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            model_params.q_a_model_specs.model_name,
            question,
            model_params.text_data,
            top_k,
        )
        for _ in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "question",
        "context_text",
        "top_k",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = QuestionAnsweringUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["answer", "score", "rank", "error_message"]

    result = Result(result_df)
    assert (
        result == ScoreMatcher()
        and result == RankDTypeMatcher()
        and result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=top_k)
        and result
        == ShapeMatcher(
            columns=columns,
            new_columns=new_columns,
            n_rows=n_rows,
            results_per_row=top_k,
        )
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NoErrorMessageMatcher()
    )


@pytest.mark.parametrize(
    "description,  device_id, n_rows, top_k",
    [
        ("on CPU with batch input, single answer", None, 3, 1),
        ("on CPU with batch input, multiple answers", None, 3, 2),
        ("on CPU with single input, single answer", None, 1, 1),
        ("on CPU with single input, multiple answers", None, 1, 2),
        ("on GPU with batch input, single answer", 0, 3, 1),
        ("on GPU with batch input, multiple answers", 0, 3, 2),
        ("on GPU with single input, single answer", 0, 1, 1),
        ("on GPU with single input, multiple answers", 0, 1, 2),
    ],
)
def test_question_answering_udf_on_error_handling(
    description,
    device_id,
    n_rows,
    top_k,
    prepare_question_answering_model_for_local_bucketfs,
):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(
            f"There is no available device({device_id}) " f"to execute the test"
        )

    bucketfs_base_path = prepare_question_answering_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    question = "Where is the Exasol?"
    sample_data = [
        (
            None,
            bucketfs_conn_name,
            model_params.sub_dir,
            "not existing model",
            question,
            model_params.text_data,
            top_k,
        )
        for _ in range(n_rows)
    ]
    columns = [
        "device_id",
        "bucketfs_conn",
        "sub_dir",
        "model_name",
        "question",
        "context_text",
        "top_k",
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = MockContext(input_df=sample_df)
    exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = QuestionAnsweringUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ["answer", "score", "rank", "error_message"]

    result = Result(result_df)
    assert (
        result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
        and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
        and result == NewColumnsEmptyMatcher(new_columns=new_columns)
        and result == ErrorMessageMatcher()
    )
