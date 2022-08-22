import pandas as pd
from typing import Dict

import pytest
import torch
from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.udfs.models.sequence_classification_single_text_udf import \
    SequenceClassificationSingleText
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
    "device_id, upload_model_to_local_bucketfs", [
        (None, model_params.base),
        (0, model_params.base)
    ], indirect=["upload_model_to_local_bucketfs"])
def test_sequence_classification_single_text_udf(
        device_id, upload_model_to_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.base,
        model_params.text_data + str(i)
    ) for i in range(n_rows)]
    sample_df = pd.DataFrame(
        data=sample_data,
        columns=[
            'device_id',
            'bucketfs_conn',
            'sub_dir',
            'model_name',
            'text_data'])

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationSingleText(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    grouped_by_inputs = result_df.groupby('text_data')
    n_unique_labels_per_input = grouped_by_inputs['label'].nunique().to_list()
    n_labels_per_input_expected = [2] * n_rows
    assert n_unique_labels_per_input == n_labels_per_input_expected \
           and result_df.shape == (n_rows*2, 6)
