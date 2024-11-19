from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    text_docid, text_start, text_end, agg_strategy_simple, make_model_output_for_one_input_row, sub_dir, model_name


class ErrorPredictionMissingExpectedFields:
    """

    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 5
    n_entities = 3

    text_data = "error_result_missing_field_'word'" #todo do we want tests for different combinations? seems like a lot of work
    # todo do we want tests for multiple models? multiple inputs where one works and one does not? how many test cases are to many test cases?
    # todo these should be moved to the base model tests together with the others

    input_data = make_input_row(text_data=text_data) * data_size
    output_data = make_output_row(text_data=text_data, score=None, error_msg="Traceback") * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span(text_data=text_data) * data_size
    work_with_span_output_data =  make_output_row_with_span(score=None,
                                                            error_msg="Traceback") * n_entities * data_size


    number_complete_batches = data_size // batch_size
    number_remaining_data_entries_in_last_batch = data_size % batch_size

    model_output_row_missing_key = [[model_output_row[0].pop("score")]
                                    for model_output_row in make_model_output_for_one_input_row(number_entities=n_entities)]

    tokenizer_model_output_df_model1 = [model_output_row_missing_key * batch_size] * \
                                number_complete_batches + \
                                [model_output_row_missing_key * number_remaining_data_entries_in_last_batch]
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }