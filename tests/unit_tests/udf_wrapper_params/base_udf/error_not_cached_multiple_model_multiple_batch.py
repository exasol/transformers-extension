from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.base_udf.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    sub_dir,  make_model_output_for_one_input_row, make_number_of_strings


class ErrorNotCachedMultipleModelMultipleBatch:
    """
    not cached error, multiple model, multiple batch
    """
    expected_model_counter = 1
    batch_size = 3
    data_size = 2

    bfs_conn1, bfs_conn2 = make_number_of_strings(bucketfs_conn, 2)
    subdir1, subdir2 = make_number_of_strings(sub_dir, 2)

    input_data = make_input_row(bucketfs_conn=bfs_conn1, sub_dir=subdir1) * data_size + \
                 make_input_row(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                                model_name="non_existing_model") * data_size

    output_data = make_output_row(bucketfs_conn=bfs_conn1, sub_dir=subdir1) * data_size + \
                  make_output_row(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                                  model_name="non_existing_model",
                                  score=None, answer=None,
                                  error_msg="Traceback") * 1 * data_size #error on load_model -> only one output per input

    work_with_span_input_data = make_input_row_with_span(bucketfs_conn=bfs_conn1, sub_dir=subdir1) * data_size + \
                                make_input_row_with_span(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                                                         model_name="non_existing_model") * data_size
    work_with_span_output_data = make_output_row_with_span(bucketfs_conn=bfs_conn1, sub_dir=subdir1) * data_size + \
                                 make_output_row_with_span(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                                                 model_name="non_existing_model",
                                                 score=None, answer=None,
                                                 error_msg="Traceback") * 1 * data_size #error on load_model -> only one output per input


    tokenizer_model_output_df_model1 =  [make_model_output_for_one_input_row() * data_size]

    tokenizer_models_output_df = [tokenizer_model_output_df_model1]


    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}")
    }

