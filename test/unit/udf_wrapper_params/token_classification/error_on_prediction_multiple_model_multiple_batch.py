from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from test.unit.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    text_docid, text_start, text_end, agg_strategy_simple, make_model_output_for_one_input_row, make_number_of_strings, sub_dir, model_name


class ErrorOnPredictionMultipleModelMultipleBatch:
    """
    not cached error, multiple model, multiple batch
    """
    expected_model_counter = 2
    batch_size = 3
    data_size = 2
    n_entities = 3

    bfs_conn1, bfs_conn2 = make_number_of_strings(bucketfs_conn, 2)
    subdir1, subdir2 = make_number_of_strings(sub_dir, 2)
    model1, model2 = make_number_of_strings(model_name, 2)

    input_data = make_input_row(bucketfs_conn=bfs_conn1, sub_dir=subdir1, model_name=model1) * data_size + \
                 make_input_row(bucketfs_conn=bfs_conn2, sub_dir=subdir2, model_name=model2,
                                text_data="error on pred") * data_size

    output_data = make_output_row(bucketfs_conn=bfs_conn1, sub_dir=subdir1, model_name=model1) * n_entities * data_size + \
                  make_output_row(bucketfs_conn=bfs_conn2, sub_dir=subdir2, model_name=model2,
                                  text_data="error on pred",
                                  score=None, start=None, end=None, word=None, entity=None,
                                  error_msg="Traceback") * 1 * data_size  #error on pred -> only one output per input

    work_with_span_input_data = make_input_row_with_span(bucketfs_conn=bfs_conn1, sub_dir=subdir1, model_name=model1) * data_size + \
                                make_input_row_with_span(bucketfs_conn=bfs_conn2, sub_dir=subdir2, model_name=model2,
                                                         text_data="error on pred") * data_size

    work_with_span_output_data1 = make_output_row_with_span(bucketfs_conn=bfs_conn1, sub_dir=subdir1, model_name=model1) * n_entities * data_size
    work_with_span_output_data2 = [(bfs_conn2, subdir2, model2, text_docid, text_start, text_end, agg_strategy_simple,
                                   None, None, None, None, None, None,"Traceback")] * 1 * data_size #error on pred -> only one output per input
    work_with_span_output_data = work_with_span_output_data1 + work_with_span_output_data2

    tokenizer_model_output_df_model1 =  [make_model_output_for_one_input_row(number_entities=n_entities) * data_size]
    tokenizer_model_output_df_model2_batch1 =  [[Exception("Traceback mock_pipeline is throwing an error intentionally")]] #error on pred -> only one output per input

    tokenizer_model_output_df_model2_batch2 =  [[Exception("Traceback mock_pipeline is throwing an error intentionally")]]#error on pred -> only one output per input

    tokenizer_models_output_df = [tokenizer_model_output_df_model1, tokenizer_model_output_df_model2_batch1, tokenizer_model_output_df_model2_batch2]


    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}")
    }
