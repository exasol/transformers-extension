import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.base_udf.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    make_model_output_for_one_input_row

from exasol_udf_mock_python.connection import Connection

@dataclasses.dataclass
class SingleModelMultipleBatchComplete:
    """
    single model, multiple batch, last batch complete
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 4

    input_data = make_input_row() * data_size
    output_data = make_output_row() * data_size

    work_with_span_input_data = make_input_row_with_span() * data_size
    work_with_span_output_data = make_output_row_with_span() * data_size

    # this is what the mock model returns to the udf
    tokenizer_model_output_df_model1 = [make_model_output_for_one_input_row() * batch_size] + \
                                       [make_model_output_for_one_input_row() * batch_size]
    tokenizer_models_output_df = [tokenizer_model_output_df_model1]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }
