import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.ai_classify_extended.make_data_row_functions import (
    LabelScore,
    LabelScores,
    bucketfs_conn,
    candidate_labels,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_with_span,
    make_udf_output_for_one_input_row_without_span,
    model_name,
    sub_dir,
)
from test.unit.utils.utils_for_udf_tests import make_number_of_strings

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class ReturnAllErrorOnPredictionMultipleModelMultipleBatch:
    """
    return_ranks ALL, not cached error, multiple models, multiple batches
    """

    expected_model_counter = 2
    batch_size = 3
    data_size = 2

    error_label_scores = LabelScores(
        [
            LabelScore(None, None, None),
            LabelScore(None, None, None),
            LabelScore(None, None, None),
            LabelScore(None, None, None),
        ]
    )

    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir, 2)
    model_name1, model_name2 = make_number_of_strings(model_name, 2)
    # create additional candidate_labels with slightly different content,
    # so it's easier to see if the output is sorted correctly to the input.
    candidate_labels2 = [
        candidate_labels[i] + str(i) for i in range(0, len(candidate_labels))
    ]

    input_data = (
        make_input_row(
            sub_dir=sub_dir1,
            model_name=model_name1,
        )
        * data_size
        + make_input_row(
            sub_dir=sub_dir2,
            model_name=model_name2,
            text_data="error on pred",
            candidate_labels=candidate_labels2,
        )
        * data_size
    )

    output_data = (
        make_udf_output_for_one_input_row_without_span(
            sub_dir=sub_dir1,
            model_name=model_name1,
        )
        * data_size
        + make_udf_output_for_one_input_row_without_span(
            sub_dir=sub_dir2,
            model_name=model_name2,
            text_data="error on pred",
            candidate_labels=candidate_labels2,
            label_scores=error_label_scores,
            error_msg="Traceback",
        )
        * data_size
    )

    zero_shot_model_output_df_batch1 = [
        make_model_output_for_one_input_row() * data_size,
        make_model_output_for_one_input_row(error_label_scores) * data_size,
    ]
    zero_shot_model_output_df_batch2 = [
        make_model_output_for_one_input_row(error_label_scores) * data_size
    ]

    zero_shot_models_output_df = [
        zero_shot_model_output_df_batch1,
        zero_shot_model_output_df_batch2,
    ]

    work_with_span_input_data = (
        make_input_row_with_span(
            sub_dir=sub_dir1,
            model_name=model_name1,
        )
        * data_size
        + make_input_row_with_span(
            sub_dir=sub_dir2,
            model_name=model_name2,
            text_data="error on pred",
            candidate_labels=candidate_labels2,
        )
        * data_size
    )

    work_with_span_output_data = (
        make_udf_output_for_one_input_row_with_span(
            sub_dir=sub_dir1,
            model_name=model_name1,
        )
        * data_size
        + make_udf_output_for_one_input_row_with_span(
            sub_dir=sub_dir2,
            model_name=model_name2,
            label_scores=error_label_scores,
            error_msg="Traceback",
        )
        * data_size
    )

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }
