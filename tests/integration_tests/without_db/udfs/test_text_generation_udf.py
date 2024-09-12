from typing import Dict

import pandas as pd
import pytest
import torch
from exasol_udf_mock_python.connection import Connection

from exasol_transformers_extension.udfs.models.text_generation_udf import \
    TextGenerationUDF
from tests.integration_tests.without_db.udfs.matcher import Result, ShapeMatcher, NewColumnsEmptyMatcher, \
    ErrorMessageMatcher, ScoreMatcher, ColumnsMatcher, NoErrorMessageMatcher
from tests.utils.parameters import model_params
from tests.utils.mock_connections import create_mounted_bucketfs_connection


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
        description, device_id, n_rows, prepare_text_generation_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_text_generation_model_for_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    batch_size = 2
    text_data = "Exasol is an analytics database management"
    max_length = 10
    return_full_text = True
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.text_gen_model_specs.model_name,
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

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
            and result == ColumnsMatcher(columns=columns[1:], new_columns=new_columns)
            and result == NoErrorMessageMatcher()
            and result_df["generated_text"].str.contains(text_data).all()
    )


@pytest.mark.parametrize(
    "description,  device_id, n_rows", [
        ("on CPU with batch input", None, 3),
        ("on CPU with single input", None, 1),
        ("on GPU with batch input", 0, 3),
        ("on GPU with single input", 0, 1)
    ])
def test_text_generation_udf_on_error_handlig(
        description, device_id, n_rows, prepare_text_generation_model_for_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = prepare_text_generation_model_for_local_bucketfs
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

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=n_rows)
            and result == NewColumnsEmptyMatcher(new_columns=new_columns)
            and result == ErrorMessageMatcher()
    )
