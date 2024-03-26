import torch
import pytest
import pandas as pd
from typing import Dict

from tests.integration_tests.without_db.udfs.matcher import Result, ShapeMatcher, NewColumnsEmptyMatcher, \
    ErrorMessageMatcher, ScoreMatcher, RankDTypeMatcher, NoErrorMessageMatcher, RankMonotonicMatcher, ColumnsMatcher
from tests.utils.parameters import model_params
from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.udfs.models.question_answering_udf import \
    QuestionAnsweringUDF


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

    def get_dataframe(self, num_rows='all', start_col=0):
        return_df = None if self._is_accessed_once \
            else self.input_df[self.input_df.columns[start_col:]]
        self._is_accessed_once = True
        return return_df


@pytest.mark.parametrize(
    "description,  device_id, n_rows, top_k", [
        ("on CPU with batch input, single answer", None, 3, 1),
        ("on CPU with batch input, multiple answers", None, 3, 2),
        ("on CPU with single input, single answer", None, 1, 1),
        ("on CPU with single input, multiple answers", None, 1, 2),
        ("on GPU with batch input, single answer", 0, 3, 1),
        ("on GPU with batch input, multiple answers", 0, 3, 2),
        ("on GPU with single input, single answer", 0, 1, 1),
        ("on GPU with single input, multiple answers", 0, 1, 2)
    ])
def test_question_answering_udf(
        description, device_id, n_rows,
        top_k, prepare_base_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_base_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    question = "Where is the Exasol?"
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.base_model,
        question,
        model_params.text_data,
        top_k
    ) for _ in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'question',
        'context_text',
        'top_k']

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = QuestionAnsweringUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['answer', 'score', 'rank', 'error_message']

    result = Result(result_df)
    assert (
            result == ScoreMatcher()
            and result == RankDTypeMatcher()
            and result == RankMonotonicMatcher(n_rows=n_rows, results_per_row=top_k)
            and result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows, results_per_row=top_k)
            and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NoErrorMessageMatcher()
    )


@pytest.mark.parametrize(
    "description,  device_id, n_rows, top_k", [
        ("on CPU with batch input, single answer", None, 3, 1),
        ("on CPU with batch input, multiple answers", None, 3, 2),
        ("on CPU with single input, single answer", None, 1, 1),
        ("on CPU with single input, multiple answers", None, 1, 2),
        ("on GPU with batch input, single answer", 0, 3, 1),
        ("on GPU with batch input, multiple answers", 0, 3, 2),
        ("on GPU with single input, single answer", 0, 1, 1),
        ("on GPU with single input, multiple answers", 0, 1, 2)
    ])
def test_question_answering_udf_on_error_handling(
        description, device_id, n_rows,
        top_k, prepare_base_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_base_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    question = "Where is the Exasol?"
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        "not existing model",
        question,
        model_params.text_data,
        top_k
    ) for _ in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'question',
        'context_text',
        'top_k']

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = QuestionAnsweringUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['answer', 'score', 'rank', 'error_message']

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
            and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NewColumnsEmptyMatcher(new_columns=new_columns)
            and result == ErrorMessageMatcher()
    )
