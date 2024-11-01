from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    make_model_output_for_one_input_row

class SingleModelMultipleBatchIncomplete:
    """
    single model, multiple batch, last batch incomplete
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 5
    n_entities = 3

    input_data = make_input_row() * data_size
    output_data = make_output_row() * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span() * data_size
    work_with_span_output_data = make_output_row_with_span()  * n_entities * data_size

    # this is what the mock model returns to the udf
    #todo make function for whole output?
    number_complete_batches = data_size // batch_size
    number_remaining_data_entries_in_last_batch = data_size % batch_size
    tokenizer_model_output_df = [make_model_output_for_one_input_row(number_entities=n_entities) * batch_size] * \
                                number_complete_batches + \
                                [make_model_output_for_one_input_row(number_entities=n_entities) * number_remaining_data_entries_in_last_batch]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, "bfs_conn1")
    bfs_connections = {
        "bfs_conn1": Connection(address=f"file://{base_cache_dir1}")
    }

