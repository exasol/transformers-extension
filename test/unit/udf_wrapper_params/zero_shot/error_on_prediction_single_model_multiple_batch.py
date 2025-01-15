from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection

from test.unit.udf_wrapper_params.zero_shot.make_data_row_functions import make_number_of_strings, sub_dir, \
    model_name, text_data, label, make_input_row, score, make_output_row, make_model_output_for_one_input_row, \
    bucketfs_conn

class ErrorOnPredictionSingleModelMultipleBatch:
    """
    not cached error, single model, multiple batch
    """
    expected_model_counter = 1
    batch_size = 2
    data_size = 5

    input_data = make_input_row(text_data="error on pred", candidate_labels=label) * data_size

    output_data = make_output_row(text_data="error on pred", candidate_labels=label,
                                  label=None,  score=None, rank=None, error_msg="Traceback") * data_size

    zero_shot_model_output_df_batch1 = []
    zero_shot_model_output_df_batch2 = []
    zero_shot_model_output_df_batch3 = []

    zero_shot_models_output_df = [zero_shot_model_output_df_batch1, zero_shot_model_output_df_batch2,
                                  zero_shot_model_output_df_batch3]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }
