from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    text_docid, text_start, text_end, agg_strategy_simple, make_model_output_for_one_input_row, sub_dir, model_name


class ErrorOnPredictionSingleModelMultipleBatch:
    """
    error on prediction, single model, multiple batch,
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 5
    n_entities = 3

    input_data = make_input_row(text_data="error on pred") * data_size
    output_data = make_output_row(text_data="error on pred",
                                  score=None, start=None, end=None, word=None, entity=None,
                                  error_msg="Traceback") * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span(text_data="error on pred") * data_size
    work_with_span_output_data =  [(bucketfs_conn, sub_dir, model_name,
                                text_docid, text_start, text_end, agg_strategy_simple,
                                None, None, None, text_docid, None, None,
                                "Traceback")] * n_entities * data_size


    number_complete_batches = data_size // batch_size
    number_remaining_data_entries_in_last_batch = data_size % batch_size
    tokenizer_model_output_df_model1 = [make_model_output_for_one_input_row(number_entities=n_entities,
                                  score=None, start=None, end=None, word=None, entity_group=None,) * batch_size] * \
                                number_complete_batches + \
                                [make_model_output_for_one_input_row(number_entities=n_entities,
                                  score=None, start=None, end=None, word=None, entity_group=None,
                                  ) * number_remaining_data_entries_in_last_batch]
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }


