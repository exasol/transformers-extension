from exasol_udf_mock_python.column import Column
from exasol_udf_mock_python.mock_meta_data import MockMetaData

from tests.unit_tests.udfs.base_model_dummy_implementation import DummyImplementationUDF
import re

class regex_matcher:
    """Assert that a given string meets some expectations."""
    def __init__(self, pattern, flags=0):
        self._regex = re.compile(pattern, flags)

    def __eq__(self, actual):
        return bool(self._regex.match(actual))

    def __repr__(self):
        return self._regex.pattern


def create_mock_metadata() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),
            Column("answer", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("error_message", str, "VARCHAR(2000000)")
        ]
    )
    return meta

def create_mock_metadata_with_span() -> MockMetaData:
    meta = MockMetaData(
        script_code_wrapper_function=None,
        input_type="SET",
        input_columns=[
            Column("device_id", int, "INTEGER"),
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),
            Column("test_span_column_drop", str, "VARCHAR(2000000)"),
        ],
        output_type="EMITS",
        output_columns=[
            Column("model_name", str, "VARCHAR(2000000)"),
            Column("sub_dir", str, "VARCHAR(2000000)"),
            Column("bucketfs_conn", str, "VARCHAR(2000000)"),
            Column("input_data", str, "VARCHAR(2000000)"),
            Column("answer", str, "VARCHAR(2000000)"),
            Column("score", float, "DOUBLE"),
            Column("test_span_column_add", str, "VARCHAR(2000000)"),
            Column("error_message", str, "VARCHAR(2000000)"),
        ]
    )
    return meta

def run_test(mock_exa, mock_base_model_factory, mock_tokenizer_factory,
             mock_pipeline, mock_ctx, batch_size=100, work_with_span=False):
    udf = DummyImplementationUDF(exa=mock_exa,
                                 base_model=mock_base_model_factory,
                                 batch_size=batch_size,
                                 tokenizer=mock_tokenizer_factory,
                                 pipeline=mock_pipeline,
                                 work_with_spans=work_with_span)
    udf.run(mock_ctx)
    res = mock_ctx.output
    return res