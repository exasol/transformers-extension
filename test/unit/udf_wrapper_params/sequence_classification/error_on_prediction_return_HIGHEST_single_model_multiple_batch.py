from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.sequence_classification.make_data_row_functions import (
    LabelScore,
    LabelScores,
    bucketfs_conn,
    make_input_row_single_text,
    make_input_row_text_pair,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_single_text,
    make_udf_output_for_one_input_row_text_pair,
)

from exasol_udf_mock_python.connection import Connection


class ErrorOnPredictionReturnHighestSingleModelMultipleBatch:
    """
    error on prediction, single model, multiple batch,
    """

    expected_single_text_model_counter = 1
    expected_text_pair_model_counter = 1
    batch_size = 2
    data_size = 5
    return_ranks = "HIGHEST"

    label_scores = LabelScores(
        [
            LabelScore(None, None, None),
            LabelScore(None, None, None),
            LabelScore(None, None, None),
            LabelScore(None, None, None),
        ]
    )

    error_msg = "Traceback"
    text_data = "error on pred"

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir}"),
    }

    inputs_single_text = (
        make_input_row_single_text(text_data=text_data, return_ranks=return_ranks)
        * data_size
    )

    outputs_single_text = (
        make_udf_output_for_one_input_row_single_text(
            text_data=text_data,
            error_msg=error_msg,
            label_scores=label_scores,
            return_ranks=return_ranks,
        )
        * data_size
    )

    sequence_models_output_df_single_text = [
        [
            make_model_output_for_one_input_row(
                label_scores=label_scores,
            )
        ]
        * data_size
    ]

    # -------------------------------------------------------

    inputs_pair_text = (
        make_input_row_text_pair(text_data_1=text_data, return_ranks=return_ranks)
        * data_size
    )

    outputs_text_pair = (
        make_udf_output_for_one_input_row_text_pair(
            text_data_1=text_data,
            error_msg=error_msg,
            label_scores=label_scores,
            return_ranks=return_ranks,
        )
        * data_size
    )

    sequence_models_output_df_text_pair = [
        [
            make_model_output_for_one_input_row(
                label_scores=label_scores,
            )
        ]
        * data_size
    ]

