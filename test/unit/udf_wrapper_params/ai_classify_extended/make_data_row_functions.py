# default values, used for input/output rows if no other params are given
from dataclasses import dataclass
from typing import Union

device_id = None  # todo rename parameters
bucketfs_conn = "bfs_conn"
sub_dir = "sub_dir"
model_name = "model"

text_data = "text"
text_doc_id = 1
text_start = 0
text_end = 6

return_ranks = "ALL"

error_msg = None


@dataclass
class LabelScore:
    label: Union[str, None]
    score: Union[float, None]
    rank: Union[int, None]


@dataclass
class LabelScores:
    label_scores: list[LabelScore]


LABEL_SCORES = LabelScores(
    [
        LabelScore("label1", 0.21, 4),
        LabelScore("label2", 0.24, 3),
        LabelScore("label3", 0.26, 2),
        LabelScore("label4", 0.29, 1),
    ]
)


def make_candidate_lables_from_lable_scores(label_scores: LabelScores = LABEL_SCORES):
    return [
        label_scores.label_scores[i].label
        for i in range(len(label_scores.label_scores))
    ]


candidate_labels = make_candidate_lables_from_lable_scores(LABEL_SCORES)


def make_input_row(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    candidate_labels=candidate_labels,
    return_ranks=return_ranks,
):
    """
    Creates an input row for zero shot classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    candidate_labels_str = ",".join(candidate_labels)
    return [
        (
            device_id,
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            candidate_labels_str,
            return_ranks,
        )
    ]


def make_udf_output_for_one_input_row_without_span(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    candidate_labels=candidate_labels,
    return_ranks=return_ranks,
    label_scores=LABEL_SCORES,
    error_msg=error_msg,
):
    """
    Creates the output row for zero shot classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    candidate_labels_str = ",".join(candidate_labels)
    if return_ranks == "ALL" and not error_msg:
        udf_output = []
        for label_score in label_scores.label_scores:
            udf_output_row = (
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data,
                candidate_labels_str,
                return_ranks,
                label_score.label,
                label_score.score,
                label_score.rank,
                error_msg,
            )

            udf_output.append(udf_output_row)

    elif return_ranks == "HIGHEST" or error_msg:
        udf_output = [
            (
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data,
                candidate_labels_str,
                return_ranks,
                label_scores.label_scores[3].label,
                label_scores.label_scores[3].score,
                label_scores.label_scores[3].rank,
                error_msg,
            )
        ]

    return udf_output


def make_model_output_for_one_input_row(label_scores: LabelScores = LABEL_SCORES):
    """
    Makes the output the model returns to the udf for one input row.
    returns a list with the model output row.
    each model output row is a dictionary.
    """

    model_output = []
    for label_score in label_scores.label_scores:
        model_output_single_label = {
            "labels": label_score.label,
            "scores": label_score.score,
        }
        model_output.append(model_output_single_label)
    return [model_output]


def make_input_row_with_span(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    text_data_doc_id=text_doc_id,
    text_data_char_begin=text_start,
    text_data_char_end=text_end,
    candidate_labels=candidate_labels,
    return_ranks=return_ranks,
):
    """
    Creates an input row for zero shot classification with span usage as a list,
    using default values for all parameters that are not specified.
    """
    candidate_labels_str = ",".join(candidate_labels)
    return [
        (
            device_id,
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            text_data_doc_id,
            text_data_char_begin,
            text_data_char_end,
            candidate_labels_str,
            return_ranks,
        )
    ]


def make_udf_output_for_one_input_row_with_span(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data_doc_id=text_doc_id,
    text_data_char_begin=text_start,
    text_data_char_end=text_end,
    return_ranks=return_ranks,
    label_scores=LABEL_SCORES,
    error_msg=error_msg,
):
    """
    Creates the output row for zero shot classification with span usage as a list,
    using default values for all parameters that are not specified.
    """
    if return_ranks == "ALL" and not error_msg:
        udf_output = []
        for label_score in label_scores.label_scores:
            udf_output_row = (
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data_doc_id,
                text_data_char_begin,
                text_data_char_end,
                return_ranks,
                label_score.label,
                label_score.score,
                label_score.rank,
                error_msg,
            )

            udf_output.append(udf_output_row)

    elif return_ranks == "HIGHEST" or error_msg:
        udf_output = [
            (
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data_doc_id,
                text_data_char_begin,
                text_data_char_end,
                return_ranks,
                label_scores.label_scores[3].label,
                label_scores.label_scores[3].score,
                label_scores.label_scores[3].rank,
                error_msg,
            )
        ]

    return udf_output
