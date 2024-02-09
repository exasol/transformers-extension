import tempfile
import torch
import pytest
import pandas as pd
from typing import Dict
from exasol_transformers_extension.udfs.models.translation_udf import \
    TranslationUDF
from tests.integration_tests.without_db.udfs.matcher import Result, ScoreMatcher, ShapeMatcher, NoErrorMessageMatcher, \
    NewColumnsEmptyMatcher, ErrorMessageMatcher
from tests.utils.parameters import model_params
from exasol_udf_mock_python.connection import Connection
from tests.fixtures.model_fixture import upload_seq2seq_model_to_local_bucketfs

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
    "description,  device_id, languages", [
        ("on CPU with single input", None, [
            ("English", "French")]),
        ("on CPU with batch input, single-pair language", None, [
            ("English", "French")] * 3),
        ("on CPU with batch input, multi language", None, [
            ("English", "French"), ("English", "German"),
            ("English", "Romanian")]),
        ("on GPU with single input", 0, [
            ("English", "French")]),
        ("on GPU with batch input, single-pair language", 0, [
            ("English", "French")] * 3),
        ("on GPU with batch input, multi language", 0, [
            ("English", "French"), ("English", "German"),
            ("English", "Romanian")])
    ])
def test_translation_udf(
        description, device_id, languages,
        upload_seq2seq_model_to_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_seq2seq_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        None,
        model_params.sub_dir,
        model_params.seq2seq_model,
        model_params.text_data,
        src_lang,
        target_lang,
        50
    ) for src_lang, target_lang in languages]
    columns = [
        'device_id',
        'bucketfs_conn',
        'token_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'source_language',
        'target_language',
        'max_length'
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TranslationUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['translation_text', 'error_message']

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=len(languages))
            and result == NoErrorMessageMatcher()
    )


@pytest.mark.parametrize(
    "description,  device_id, languages", [
        ("on CPU with single input", None, [
            ("English", "French")]),
        ("on CPU with batch input, single-pair language", None, [
            ("English", "French")] * 3),
        ("on CPU with batch input, multi language", None, [
            ("English", "French"), ("English", "German"),
            ("English", "Romanian")]),
        ("on GPU with single input", 0, [
            ("English", "French")]),
        ("on GPU with batch input, single-pair language", 0, [
            ("English", "French")] * 3),
        ("on GPU with batch input, multi language", 0, [
            ("English", "French"), ("English", "German"),
            ("English", "Romanian")])
    ])
def test_translation_udf_on_error_handling(
        description, device_id, languages,
        upload_seq2seq_model_to_local_bucketfs):
    if device_id is not None and not torch.cuda.is_available():
        pytest.skip(f"There is no available device({device_id}) "
                    f"to execute the test")

    bucketfs_base_path = upload_seq2seq_model_to_local_bucketfs
    bucketfs_conn_name = "bucketfs_connection"
    bucketfs_connection = Connection(address=f"file://{bucketfs_base_path}")

    batch_size = 2
    sample_data = [(
        None,
        bucketfs_conn_name,
        None,
        model_params.sub_dir,
        "not existing model",
        model_params.text_data,
        src_lang,
        target_lang,
        50
    ) for src_lang, target_lang in languages]
    columns = [
        'device_id',
        'bucketfs_conn',
        'token_conn',
        'sub_dir',
        'model_name',
        'text_data',
        'source_language',
        'target_language',
        'max_length'
    ]

    sample_df = pd.DataFrame(data=sample_data, columns=columns)
    ctx = Context(input_df=sample_df)
    exa = ExaEnvironment({bucketfs_conn_name: bucketfs_connection})

    sequence_classifier = TranslationUDF(
        exa, batch_size=batch_size)
    sequence_classifier.run(ctx)

    result_df = ctx.get_emitted()[0][0]
    new_columns = ['translation_text', 'error_message']

    result = Result(result_df)
    assert (
            result == ShapeMatcher(columns=columns, new_columns=new_columns, n_rows=len(languages))
            and result == NewColumnsEmptyMatcher(new_columns=new_columns)
            and result == ErrorMessageMatcher()
    )
