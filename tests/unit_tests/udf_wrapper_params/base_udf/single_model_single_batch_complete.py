from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.base_udf.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    make_model_output_for_one_input_row


class SingleModelSingleBatchComplete:
    """
    single model, single batch, batch complete
    """
    expected_model_counter = 1
    batch_size = 1
    data_size = 1
    n_entities = 1
    input_data = make_input_row() * data_size
    output_data = make_output_row() * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span() * data_size
    work_with_span_output_data = make_output_row_with_span()  * n_entities * data_size

    # this is what the mock model returns to the udf
    tokenizer_model_output_df_model1 = [make_model_output_for_one_input_row(number_entities=n_entities) * batch_size]
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }


