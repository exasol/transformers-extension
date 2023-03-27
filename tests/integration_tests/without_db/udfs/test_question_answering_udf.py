import torch
import pytest
import pandas as pd
from typing import Dict
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
        top_k, upload_base_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_base_model_to_local_bucketfs
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

    # assertions
    is_score_typed_as_float = result_df['score'].dtypes == 'float'
    is_rank_typed_as_int = result_df['rank'].dtypes == 'int'
    is_error_message_none = not any(result_df['error_message'])
    has_valid_shape =  \
        result_df.shape == (n_rows * top_k, len(columns)+len(new_columns)-1)
    has_valid_column_number = \
        result_df.shape[1] == len(columns) + len(new_columns) - 1
    is_rank_correct = \
        all([result_df[row*top_k: top_k + row*top_k]
            .sort_values(by='score', ascending=False)['rank']
            .is_monotonic for row in range(n_rows)])

    assert all((
        is_score_typed_as_float,
        is_rank_typed_as_int,
        is_error_message_none,
        has_valid_shape,
        has_valid_column_number,
        is_rank_correct
    ))


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
        top_k, upload_base_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_base_model_to_local_bucketfs
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

    # assertions
    are_new_columns_none = all(
        all(result_df[col].isnull()) for col in new_columns[:-1])
    has_valid_error_message = all(
        'Traceback' in row for row in result_df['error_message'])
    has_valid_shape =  \
        result_df.shape == (n_rows, len(columns)+len(new_columns)-1)
    has_valid_column_number = \
        result_df.shape[1] == len(columns) + len(new_columns) - 1

    assert all((
        are_new_columns_none,
        has_valid_error_message,
        has_valid_shape,
        has_valid_column_number,
    ))
