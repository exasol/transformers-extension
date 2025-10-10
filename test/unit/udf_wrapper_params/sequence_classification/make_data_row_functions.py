# default values, used for input/output rows if no other params are given
from dataclasses import dataclass
from typing import Union

device_id = None
bucketfs_conn = "bfs_conn"
sub_dir = "sub_dir"
model_name = "model"

text_data = "My test text"
text_data_2 = "My test text 2"


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

return_ranks = "ALL"

error_msg = None


def make_input_row_single_text(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    return_ranks=return_ranks,
):
    """
    Creates an input row for sequence classification single text usage as a list,
    using default values for all parameters that are not specified.
    """
    return [(device_id, bucketfs_conn, sub_dir, model_name, text_data, return_ranks)]


def make_input_row_text_pair(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data_1=text_data,
    text_data_2=text_data_2,
    return_ranks=return_ranks,
):
    """
    Creates an input row for sequence classification text pair usage as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (
            device_id,
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data_1,
            text_data_2,
            return_ranks,
        )
    ]


def make_udf_output_for_one_input_row_single_text(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    return_ranks=return_ranks,
    label_scores=LABEL_SCORES,
    error_msg=error_msg,
):
    """
    Makes the output the udf should return one input row.
    depending on how return_ranks is specified.
    each model output row is a dictionary.
    """
    if return_ranks == "ALL" and not error_msg:
        udf_output = []
        for label_score in label_scores.label_scores:
            udf_output_row = (
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data,
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
                return_ranks,
                label_scores.label_scores[
                    3
                ].label,  # todo what do if not default input?
                label_scores.label_scores[3].score,
                label_scores.label_scores[3].rank,
                error_msg,
            )
        ]

    return udf_output


def make_model_output_for_one_input_row(
    label_scores=LABEL_SCORES,
):
    """
    Makes the output the model returns to the udf for one input row.
    each model output row is a dictionary.
    """
    model_output = []
    for label_score in label_scores.label_scores:
        model_output.append({"label": label_score.label, "score": label_score.score})

    return [model_output]


def make_udf_output_for_one_input_row_text_pair(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data_1=text_data,
    text_data_2=text_data_2,
    return_ranks=return_ranks,
    label_scores=LABEL_SCORES,
    error_msg=error_msg,
):
    """
    Makes the output the udf should return for one input row.
    depending on how return_ranks is specified.
    each model output row is a dictionary.
    """
    if return_ranks == "ALL" and not error_msg:
        udf_output = []
        for label_score in label_scores.label_scores:
            udf_output.append(
                (
                    bucketfs_conn,
                    sub_dir,
                    model_name,
                    text_data_1,
                    text_data_2,
                    return_ranks,
                    label_score.label,
                    label_score.score,
                    label_score.rank,
                    error_msg,
                )
            )

    elif return_ranks == "HIGHEST" or error_msg:
        # if there was an error during prediction,
        # only one result with traceback gets returned per input,
        # because the rank cant be computed
        # todo what do if not default input of label score. do i really need another sorting?
        udf_output = [
            (
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data_1,
                text_data_2,
                return_ranks,
                label_scores.label_scores[3].label,
                label_scores.label_scores[3].score,
                label_scores.label_scores[3].rank,
                error_msg,
            )
        ]
    return udf_output
