from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection

from tests.unit_tests.udf_wrapper_params.zero_shot.make_data_row_functions import make_number_of_strings, sub_dir, \
    model_name, text_data, label, make_input_row, score, make_output_row, make_model_output_for_one_input_row, \
    bucketfs_conn

class ErrorOnPredictionMultipleModelMultipleBatch:
    """
    not cached error, multiple model, multiple batch
    """
    expected_model_counter = 2
    batch_size = 3
    data_size = 2

    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir, 2)
    model_name1, model_name2 = make_number_of_strings(model_name, 2)
    label1, label2 = make_number_of_strings(label, 2)

    input_data = make_input_row(sub_dir=sub_dir1, model_name=model_name1,
                                text_data=text_data, candidate_labels=label1) * data_size + \
                 make_input_row(sub_dir=sub_dir2, model_name=model_name2,
                                text_data="error on pred", candidate_labels=label2) * data_size

    output_data = make_output_row(sub_dir=sub_dir1, model_name=model_name1,
                                  text_data=text_data, candidate_labels=label1,
                                  label=label1, score=score) * data_size + \
                  make_output_row(sub_dir=sub_dir2, model_name=model_name2,
                                  text_data="error on pred", candidate_labels=label2,
                                  label=None,  score=None, rank=None, error_msg="Traceback") * data_size

    zero_shot_model_output_df_batch1 = [make_model_output_for_one_input_row(candidate_labels=label1, score=score) * data_size,
                                       []]
    zero_shot_model_output_df_batch2 = []

    zero_shot_models_output_df = [zero_shot_model_output_df_batch1, zero_shot_model_output_df_batch2]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }