import pandas as pd
from typing import Dict
import pytest
import torch
from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.udfs.models.zero_shot_text_classification_udf import \
    ZeroShotTextClassificationUDF
from tests.utils.parameters import model_params


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
    "description, device_id", [
        ("on CPU", None),
        ("on GPU", 0)
    ])
def test_sequence_classification_single_text_udf(
        description, device_id, upload_base_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_base_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.base_model,
        model_params.text_data + str(i),
        candidate_labels + str(i)
    ) for i in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'candidate_labels']
    sample_df = pd.DataFrame( data=sample_data, columns=columns)

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = ZeroShotTextClassificationUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['label', 'score', 'rank', 'error_message']

    # assertions
    n_labels = len(candidate_labels.split(","))
    grouped_by_inputs = result_df.groupby('text_data')
    n_unique_labels_per_input = grouped_by_inputs['label'].nunique().to_list()

    is_error_message_none = not any(result_df['error_message'])
    has_valid_shape = \
        result_df.shape == (sum(n_unique_labels_per_input),
                            len(columns)+len(new_columns)-1)
    has_valid_label_number =  \
        n_unique_labels_per_input == [n_labels] * n_rows
    is_rank_correct = \
        all([result_df[row*n_labels: n_labels + row*n_labels]
            .sort_values(by='score', ascending=False)['rank']
            .is_monotonic for row in range(n_rows)])

    assert all((
        is_error_message_none,
        has_valid_shape,
        has_valid_label_number,
        is_rank_correct
    ))


@pytest.mark.parametrize(
    "description, device_id", [
        ("on CPU", None),
        ("on GPU", 0)
    ])
def test_sequence_classification_single_text_udf_on_error_handling(
        description, device_id, upload_base_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_base_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        "not existing model",
        model_params.text_data + str(i),
        candidate_labels + str(i)
    ) for i in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'candidate_labels']
    sample_df = pd.DataFrame( data=sample_data, columns=columns)

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = ZeroShotTextClassificationUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['label', 'score', 'rank', 'error_message']

    # assertions
    print(result_df.shape)
    are_new_columns_none = all(
        all(result_df[col].isnull()) for col in new_columns[:-1])
    has_valid_error_message = all(
        'Traceback' in row for row in result_df['error_message'])
    has_valid_shape = \
        result_df.shape == (n_rows, len(columns) + len(new_columns) - 1)
    has_valid_column_number = \
        result_df.shape[1] == len(columns) + len(new_columns) - 1
    assert all((
        are_new_columns_none,
        has_valid_error_message,
        has_valid_shape,
        has_valid_column_number,
    ))

