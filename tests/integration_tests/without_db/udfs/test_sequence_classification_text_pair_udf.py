import pandas as pd
from typing import Dict
import pytest
import torch
from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.udfs.models.sequence_classification_text_pair_udf import \
    SequenceClassificationTextPairUDF
from tests.integration_tests.without_db.udfs.matcher import Result, ShapeMatcher, NewColumnsEmptyMatcher, \
    ErrorMessageMatcher, ColumnsMatcher, NoErrorMessageMatcher
from tests.utils.parameters import model_params
from tests.utils.mounted_bucketfs_connection import create_mounted_bucketfs_connection


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
def test_sequence_classification_text_pair_udf(
        description, device_id, prepare_base_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_base_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    n_rows = 3
    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.base_model,
        model_params.text_data + str(i),
        model_params.text_data + str(i * i)) for i in range(n_rows)]

    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'first_text',
        'second_text']
    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationTextPairUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['label', 'score', 'error_message']

    grouped_by_inputs = result_df.groupby('first_text')
    n_unique_labels_per_input = grouped_by_inputs['label'].nunique().to_list()
    n_labels = 2
    n_labels_per_input_expected = [n_labels] * n_rows

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows, results_per_row=n_labels)
            and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NoErrorMessageMatcher()
            and n_unique_labels_per_input == n_labels_per_input_expected
    )


@pytest.mark.parametrize(
    "description, device_id", [
        ("on CPU", None),
        ("on GPU", 0)
    ])
def test_sequence_classification_text_pair_udf_on_error_handling(
        description, device_id, prepare_base_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_base_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        "not existing model",
        model_params.text_data + str(i),
        model_params.text_data + str(i * i)) for i in range(n_rows)]

    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'first_text',
        'second_text']
    sample_df = pd.DataFrame(data=sample_data, columns=columns)

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationTextPairUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['label', 'score', 'error_message']

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
            and result == NewColumnsEmptyMatcher(new_columns=new_columns)
            and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == ErrorMessageMatcher()
    )
