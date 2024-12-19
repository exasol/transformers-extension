from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.base_udf.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    sub_dir, model_name, make_model_output_for_one_input_row, make_number_of_strings


class ErrorNotCachedSingleModelMultipleBatch:
    """
    not cached error, single model, multiple batch#todo
    """
    expected_model_counter = 0
    batch_size = 2
    data_size = 5
    n_entities = 1

    input_data = make_input_row(model_name="non_existing_model") * data_size
    output_data = make_output_row(model_name="non_existing_model",
                                  score=None, answer=None,
                                  error_msg="Traceback") * 1 * data_size #error on load_model -> only one output per input

    work_with_span_input_data = make_input_row_with_span(model_name="non_existing_model") * data_size

    work_with_span_output_data = make_output_row_with_span(model_name="non_existing_model",
                                                           score=None, answer=None,
                                                           error_msg="Traceback") * 1 * data_size #error on load_model -> only one output per input


    tokenizer_models_output_df = [] # no model loaded so no model to output anything

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }


