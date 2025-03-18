import dataclasses
from pathlib import PurePosixPath

from test.unit.udf_wrapper_params.zero_shot.make_data_row_functions import label, make_input_row, make_output_row, \
    bucketfs_conn, make_input_row_with_span, make_output_row_with_span

from exasol_udf_mock_python.connection import Connection

@dataclasses.dataclass
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

    work_with_span_input_data = make_input_row_with_span(text_data="error on pred", candidate_labels=label) * data_size

    work_with_span_output_data = make_output_row_with_span(label=None,  score=None, rank=None,
                                                           error_msg="Traceback") * data_size

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }
