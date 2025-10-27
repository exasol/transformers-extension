import dataclasses
from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.zero_shot.make_data_row_functions import (
    LABEL_SCORES,
    LabelScore,
    LabelScores,
    bucketfs_conn,
    make_candidate_lables_from_lable_scores,
    make_input_row,
    make_input_row_with_span,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_with_span,
    make_udf_output_for_one_input_row_without_span,
    model_name,
    sub_dir,
    text_data,
)
from test.unit.utils.utils_for_udf_tests import make_number_of_strings

from exasol_udf_mock_python.connection import Connection


@dataclasses.dataclass
class ReturnHighestMultipleModelMultipleBatchComplete:
    """
    return_ranks HIGHEST, multiple model, multiple batches, last batch complete
    """

    expected_model_counter = 2
    batch_size = 2
    data_size = 2
    return_ranks = "HIGHEST"
    error_label_scores = LABEL_SCORES

    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir, 2)
    model_name1, model_name2 = make_number_of_strings(model_name, 2)
    text_data1, text_data2 = make_number_of_strings(text_data, 2)

    label_scores1 = LABEL_SCORES
    label_scores2 = LabelScores(
        [
            LabelScore("label21", 0.221, 4),
            LabelScore("label22", 0.224, 3),
            LabelScore("label23", 0.226, 2),
            LabelScore("label24", 0.229, 1),
        ]
    )

    input_data = (
        make_input_row(
            sub_dir=sub_dir1,
            model_name=model_name1,
            text_data=text_data1,
            candidate_labels=make_candidate_lables_from_lable_scores(label_scores1),
            return_ranks=return_ranks,
        )
        * data_size
        + make_input_row(
            sub_dir=sub_dir2,
            model_name=model_name2,
            text_data=text_data2,
            candidate_labels=make_candidate_lables_from_lable_scores(label_scores2),
            return_ranks=return_ranks,
        )
        * data_size
    )

    output_data = (
        make_udf_output_for_one_input_row_without_span(
            sub_dir=sub_dir1,
            model_name=model_name1,
            text_data=text_data1,
            label_scores=label_scores1,
            candidate_labels=make_candidate_lables_from_lable_scores(label_scores1),
            return_ranks=return_ranks,
        )
        * data_size
        + make_udf_output_for_one_input_row_without_span(
            sub_dir=sub_dir2,
            model_name=model_name2,
            text_data=text_data2,
            label_scores=label_scores2,
            candidate_labels=make_candidate_lables_from_lable_scores(label_scores2),
            return_ranks=return_ranks,
        )
        * data_size
    )

    work_with_span_input_data = (
        make_input_row_with_span(
            sub_dir=sub_dir1,
            model_name=model_name1,
            text_data=text_data1,
            candidate_labels=make_candidate_lables_from_lable_scores(label_scores1),
            return_ranks=return_ranks,
        )
        * data_size
        + make_input_row_with_span(
            sub_dir=sub_dir2,
            model_name=model_name2,
            text_data=text_data2,
            candidate_labels=make_candidate_lables_from_lable_scores(label_scores2),
            return_ranks=return_ranks,
        )
        * data_size
    )

    work_with_span_output_data = (
        make_udf_output_for_one_input_row_with_span(
            sub_dir=sub_dir1,
            model_name=model_name1,
            label_scores=label_scores1,
            return_ranks=return_ranks,
        )
        * data_size
        + make_udf_output_for_one_input_row_with_span(
            sub_dir=sub_dir2,
            model_name=model_name2,
            label_scores=label_scores2,
            return_ranks=return_ranks,
        )
        * data_size
    )

    zero_shot_model_output_df_model1 = [
        make_model_output_for_one_input_row(label_scores1) * data_size
    ]
    zero_shot_model_output_df_model2 = [
        make_model_output_for_one_input_row(label_scores2) * data_size
    ]

    zero_shot_models_output_df = [
        zero_shot_model_output_df_model1,
        zero_shot_model_output_df_model2,
    ]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}"),
    }
