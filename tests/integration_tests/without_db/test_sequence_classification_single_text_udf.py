import pandas as pd
from typing import Dict
from exasol_udf_mock_python.connection import Connection
from exasol_transformers_extension.udfs.sequence_classification_single_text_udf import \
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

    def get_emitted(self):
        return self._emitted

    def get_dataframe(self, num_rows='all'):
        return_df = None if self._is_accessed_once else self.input_df
        self._is_accessed_once = True
        return return_df


def test_sequence_classification_single_text_udf(
        upload_model_to_local_bucketfs):

    bucketfs_base_path = upload_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    n_rows = 3
    batch_size = 2
    sample_data = [(bucketfs_conn_name, model_params.sub_dir, model_params.name,
                    model_params.text_data + str(i)) for i in range(n_rows)]
    sample_df = pd.DataFrame(
        data=sample_data,
        columns=['bucketfs_conn', 'sub_dir', 'model_name', 'text_data'])

    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationSingleText(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    assert result_df.groupby('text_data')['label'].nunique().to_list() == \
           [2] * n_rows and result_df.shape == (n_rows*2, 6)
