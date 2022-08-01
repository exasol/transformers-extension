from typing import Dict
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from exasol_transformers_extension.udfs.question_answering_udf import \
    QuestionAnswering
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

    def emit(self, *args):
        self._emitted.append(args)

    def reset(self):
        pass

    def get_emitted(self):
        return self._emitted

    def get_dataframe(self, num_rows='all', start_col=0):
        return_df = self.input_df[self.input_df.columns[start_col:]] \
            if num_rows > 0 else None
        return return_df


class MockedTorchLibrary:
    @staticmethod
    def is_available():
        return True


@patch('torch.cuda', MagicMock(return_value=MockedTorchLibrary))
@pytest.mark.parametrize("device_id, expected",
                         [(None, "cpu"), (0, "cuda:0"), (1, "cuda:1")])
def test_device_setting(device_id, expected):
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://test_path")

    n_rows = 3
    question = "Where is the Exasol?"
    sample_data = [(
        device_id,
        bucketfs_conn_name,
        model_params.sub_dir,
        model_params.name,
        question,
        model_params.text_data
    ) for _ in range(n_rows)]
    columns = [
        'device_id',
        'bucketfs_conn',
        'sub_dir',
        'model_name',
        'question',
        'context_text']

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = QuestionAnswering(exa, batch_size=0)
    sequence_classifier.run(ctx)

    device = sequence_classifier.device
    device_name = f"{device.type}:{device.index}" \
        if device.index is not None else device.type
    assert device_name == expected


