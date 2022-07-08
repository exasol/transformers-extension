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
    def __init__(self, bucketfs_conn: str, model_name: str, text: str):
        self.bucketfs_conn = bucketfs_conn
        self.model_name = model_name
        self.text = text
        self._emitted = []

    def emit(self, *args):
        self._emitted.append(args)

    def get_emitted(self):
        return self._emitted


def test_sequence_classification_single_text_udf(
        upload_model_to_local_bucketfs):

    model_path = str(upload_model_to_local_bucketfs)
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file:///{model_path}")

    ctx = Context(
        bucketfs_conn_name,
        model_params.name,
        model_params.text)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = SequenceClassificationSingleText(
        exa, cache_dir=model_path)
    sequence_classifier.run(ctx)
    assert ctx.get_emitted()[0][0]
