import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.token_classification.make_data_row_functions import (
    bucketfs_conn,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_output_row,
    make_output_row_with_span,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class PredictionContainsAdditionalFields:
    """
    Output from model contains additional unrecognized columns.
    These are ignored and expected columns returned as normal.
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 2
    n_entities = 3

    text_data = "result contains additional keys"

    # todod these are not filled out
    input_data = make_input_row(text_data=text_data) * data_size
    output_data = make_output_row(text_data=text_data) * n_entities * data_size

    work_with_span_input_data = (
        make_input_row_with_span(text_data=text_data) * data_size
    )
    work_with_span_output_data = make_output_row_with_span() * n_entities * data_size

    model_output_rows = make_model_output_for_one_input_row(number_entities=n_entities)
    for model_output_row in model_output_rows:
        model_output_row[0].update({"unknown key": "some value", "diff unknown key": 1})

    tokenizer_model_output_df_model1 = [model_output_rows * data_size]
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")}
