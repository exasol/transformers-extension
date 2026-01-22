import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_classify_extended.make_data_row_functions import (
    LabelScore,
    LabelScores,
    bucketfs_conn,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_with_span,
    make_udf_output_for_one_input_row_without_span,
)

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class ReturnAllErrorOnPredictionSingleModelMultipleBatch:
    """
    return_ranks ALL, not cached error, single model, multiple batches
    """

    expected_model_counter = 1
    batch_size = 2
    data_size = 5

    label_scores = LabelScores(
        [
            LabelScore(None, None, None),
            LabelScore(None, None, None),
            LabelScore(None, None, None),
            LabelScore(None, None, None),
        ]
    )

    input_data = make_input_row(text_data="error on pred") * data_size

    output_data = (
        make_udf_output_for_one_input_row_without_span(
            text_data="error on pred",
            label_scores=label_scores,
            error_msg="Traceback",
        )
        * data_size
    )

    zero_shot_model_output_df_one_full_batch = (
        make_model_output_for_one_input_row(label_scores) * batch_size
    )
    zero_shot_model_output_df_last_batch = make_model_output_for_one_input_row(
        label_scores
    )

    zero_shot_models_output_df = [
        zero_shot_model_output_df_one_full_batch,
        zero_shot_model_output_df_one_full_batch,
        zero_shot_model_output_df_last_batch,
    ]

    work_with_span_input_data = (
        make_input_row_with_span(text_data="error on pred") * data_size
    )

    work_with_span_output_data = (
        make_udf_output_for_one_input_row_with_span(
            label_scores=label_scores, error_msg="Traceback"
        )
        * data_size
    )

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }
