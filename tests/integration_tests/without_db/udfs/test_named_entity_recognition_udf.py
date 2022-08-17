import pandas as pd
from typing import Dict
import pytest
import torch

from exasol_transformers_extension.udfs.models.named_entity_recognition_udf import \
    NamedEntityRecognitionUDF
from tests.utils.parameters import model_params
from exasol_udf_mock_python.connection import Connection


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


@pytest.mark.parametrize("description,  device_id, n_rows", [
    ("on CPU with batch input", None, 3),
    ("on CPU with single input", None, 1),
    ("on GPU with batch input", 0, 3),
    ("on GPU with single input", 0, 1)])
def test_text_generation_udf(
        description, device_id, n_rows, upload_model_to_local_bucketfs):

    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.name,
        model_params.text_data * (i+1),
    ) for i in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'text_data']

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = NamedEntityRecognitionUDF(exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['word_index', 'word', 'entity', 'score']
    assert result_df.shape[1] == len(columns) + len(new_columns) - 1 \
           and list(result_df.columns) == columns[1:] + new_columns
