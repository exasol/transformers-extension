from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    text_doc_id, text_start, text_end, agg_strategy_simple, make_model_output_for_one_input_row, sub_dir, model_name


class PredictionReturnsEmptyResult:
    """
    Output from model is empty. Respective input row should be dropped and remaining output returned normally.
    Tests different formats for empty result.
    """
    expected_model_counter = 1
    batch_size = 6
    data_size = 1
    n_entities = 3

    text_data = "error_result_empty"
    input_data = make_input_row() * data_size  + \
                 make_input_row(text_data=text_data) * data_size  + \
                  make_input_row(text_data=text_data) * data_size + \
                  make_input_row(text_data=text_data) * data_size + \
                  make_input_row(text_data=text_data) * data_size + \
                  make_input_row() * data_size
    output_data = make_output_row() * n_entities * data_size + \
                  make_output_row() * n_entities * data_size # Result of input #2 is empty, so the row does not appear in the output

    work_with_span_input_data = make_input_row_with_span() * data_size  + \
                                make_input_row_with_span(text_data=text_data) * data_size  + \
                                make_input_row_with_span(text_data=text_data) * data_size + \
                                make_input_row_with_span(text_data=text_data) * data_size + \
                                make_input_row_with_span(text_data=text_data) * data_size + \
                                make_input_row_with_span() * data_size

    work_with_span_output_data =  make_output_row_with_span() * n_entities * data_size  + \
                                  make_output_row_with_span() * n_entities * data_size # Result of input #2 is empty, so the row does not appear in the output


    tokenizer_model_output_df_model1 = make_model_output_for_one_input_row(number_entities=n_entities) * data_size
    tokenizer_model_output_df_model1.append([])
    tokenizer_model_output_df_model1.append({})
    tokenizer_model_output_df_model1.append([[]])
    tokenizer_model_output_df_model1.append([{}])
    tokenizer_model_output_df_model1 = tokenizer_model_output_df_model1 + make_model_output_for_one_input_row(number_entities=n_entities) * data_size

    tokenizer_models_output_df = [[tokenizer_model_output_df_model1]]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }