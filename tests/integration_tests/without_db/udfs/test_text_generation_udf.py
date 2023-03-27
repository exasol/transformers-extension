import torch
import pytest
import pandas as pd
from typing import Dict
from tests.utils.parameters import model_params
from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.udfs.models.text_generation_udf import \
    TextGenerationUDF


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
    "description,  device_id, n_rows", [
        ("on CPU with batch input", None, 3),
        ("on CPU with single input", None, 1),
        ("on GPU with batch input", 0, 3),
        ("on GPU with single input", 0, 1)
    ])
def test_text_generation_udf(
        description, device_id, n_rows, upload_base_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_base_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    text_data = "Exasol is an analytics database management"
    max_length = 10
    return_full_text = True
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.base_model,
        text_data,
        max_length,
        return_full_text
    ) for _ in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'max_length',
        'return_full_text']

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TextGenerationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['generated_text', 'error_message']

    is_error_message_none = not any(result_df['error_message'])
    has_valid_generated_text = \
        result_df["generated_text"].str.contains(text_data).all()
    has_valid_shape = \
        result_df.shape == (n_rows, len(columns)+len(new_columns)-1)
    has_valid_column_number = \
        result_df.shape[1] == len(columns) + len(new_columns) - 1

    assert all((
        is_error_message_none,
        has_valid_generated_text,
        has_valid_shape,
        has_valid_column_number
    ))


@pytest.mark.parametrize(
    "description,  device_id, n_rows", [
        ("on CPU with batch input", None, 3),
        ("on CPU with single input", None, 1),
        ("on GPU with batch input", 0, 3),
        ("on GPU with single input", 0, 1)
    ])
def test_text_generation_udf_on_error_handlig(
        description, device_id, n_rows, upload_base_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_base_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    text_data = "Exasol is an analytics database management"
    max_length = 10
    return_full_text = True
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        "not existing model",
        text_data,
        max_length,
        return_full_text
    ) for _ in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'max_length',
        'return_full_text']

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TextGenerationUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['generated_text', 'error_message']

    # assertions
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

