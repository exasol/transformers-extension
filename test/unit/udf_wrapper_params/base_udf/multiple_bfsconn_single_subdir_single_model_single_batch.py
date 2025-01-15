from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from test.unit.udf_wrapper_params.base_udf.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    answer, score, make_model_output_for_one_input_row, make_number_of_strings


class MultipleBucketFSConnSingleSubdirSingleModelNameSingleBatch:
    """
    multiple bucketfs connection, single subdir, single model, single batch
    """
    expected_model_counter = 2
    batch_size = 4
    data_size = 2

    bfs_conn1, bfs_conn2 = make_number_of_strings(bucketfs_conn, 2)
    answer1, answer2 = make_number_of_strings(answer, 2)

    input_data = make_input_row(bucketfs_conn=bfs_conn1) * data_size + \
                 make_input_row(bucketfs_conn=bfs_conn2) * data_size
    output_data = make_output_row(bucketfs_conn=bfs_conn1, answer=answer1, score=score) * data_size + \
                  make_output_row(bucketfs_conn=bfs_conn2, answer=answer2, score=score+0.1) * data_size

    work_with_span_input_data = make_input_row_with_span(bucketfs_conn=bfs_conn1) * data_size + \
                                make_input_row_with_span(bucketfs_conn=bfs_conn2) * data_size
    work_with_span_output_data =  make_output_row_with_span(bucketfs_conn=bfs_conn1, answer=answer1, score=score) * data_size + \
                                  make_output_row_with_span(bucketfs_conn=bfs_conn2, answer=answer2, score=score+0.1) * data_size

    tokenizer_model_output_df_model1 =  [make_model_output_for_one_input_row(answer=answer1, score=score) * data_size]
    tokenizer_model_output_df_model2 =  [make_model_output_for_one_input_row(answer=answer2, score=score+0.1) * data_size]

    tokenizer_models_output_df = [tokenizer_model_output_df_model1, tokenizer_model_output_df_model2]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}")
    }
