# default values, used for input/output rows if no other params are given
from dataclasses import dataclass
from typing import Union

device_id = None  # todo rename variables
bucketfs_conn = "bfs_conn"
sub_dir = "sub_dir"
model_name = "model"

text_data = "My test text"
text_data_2 = "My test text 2"

@dataclass
class LabelScore:
    label: Union[str, None]
    score: Union[float, None]

@dataclass
class LabelScores:
    label_scores: list[LabelScore]

LABEL_SCORES = LabelScores(
        [
            LabelScore("label1", 0.21),
            LabelScore("label2", 0.24),
            LabelScore("label3", 0.26),
            LabelScore("label4", 0.29)
        ])

return_rank = "ALL"

error_msg = None


def make_input_row_single_text(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    return_rank=return_rank
):
    """
    Creates an input row for sequence classification single text usage as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (device_id, bucketfs_conn, sub_dir, model_name, text_data, return_rank)
    ]

def make_input_row_text_pair(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data_1=text_data,
    text_data_2=text_data_2,
    return_rank=return_rank
):
    """
    Creates an input row for sequence classification text pair usage as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (device_id, bucketfs_conn, sub_dir, model_name, text_data_1, text_data_2, return_rank)
    ]


def make_output_row_single_text(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    return_rank=return_rank,
    label=LABEL_SCORES.label_scores[3].label,#defaults are highest score
    score=LABEL_SCORES.label_scores[3].score,
    error_msg=error_msg,
):
    """
    Creates an output row for sequence classification as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            return_rank,
            label,
            score,
            error_msg,
        )
    ]


def make_model_output_for_one_input_row_single_text(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    return_rank=return_rank,
    label_scores=LABEL_SCORES,
    error_msg=error_msg,
):
    """
    Makes the output the model returns to the udf for one input row.
    depending on how return_rank is specified.
    Unless aggregation_strategy == "none", then the type/class of the found
    token is called "entity" in the model output.
    returns a list of number_entities times the model output row.
    each model output row is a dictionary.
    """
    if return_rank == "ALL":
        model_output = []
        for label_score in label_scores.label_scores:
            model_output.append(make_output_row_single_text(
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data,
                return_rank,
                label_score.label,
                label_score.score,
                error_msg)
            )

    elif return_rank == "HIGHEST":
        model_output = [
            make_output_row_single_text(
                bucketfs_conn=bucketfs_conn,
                sub_dir=sub_dir,
                model_name=model_name,
                text_data=text_data,
                return_rank=return_rank,
                label=label_scores.label_scores[3].label,#todo what do if not default input?
                score=label_scores.label_scores[3].score,
                error_msg=error_msg)
        ]
    return model_output


def make_output_row_text_pair(#todo this should not include the inputs! just model outputs
        bucketfs_conn=bucketfs_conn,
        sub_dir=sub_dir,
        model_name=model_name,
        text_data1=text_data,
        text_data2=text_data_2,
        return_rank=return_rank,
        label=LABEL_SCORES.label_scores[3].label,  # defaults are highest score
        score=LABEL_SCORES.label_scores[3].score,
        error_msg=error_msg,
):
    """
    Creates an output row for sequence classification as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data1,
            text_data2,
            return_rank,
            label,
            score,
            error_msg,
        )
    ]


def make_model_output_for_one_input_row_text_pair(#todo this should not include the inputs! just model outputs
        bucketfs_conn=bucketfs_conn,#todo rename this to expected output, and make second one for model output. also for single text
        sub_dir=sub_dir,
        model_name=model_name,
        text_data_1=text_data,
        text_data_2=text_data_2,
        return_rank=return_rank,
        label_scores=LABEL_SCORES,
        error_msg=error_msg,
):
    """
    Makes the output the model returns to the udf for one input row.
    depending on how return_rank is specified.
    Unless aggregation_strategy == "none", then the type/class of the found
    token is called "entity" in the model output.
    returns a list of number_entities times the model output row.
    each model output row is a dictionary.
    """
    if return_rank == "ALL":
        model_output = []
        for label_score in label_scores.label_scores:
            model_output.append(make_output_row_text_pair(
                bucketfs_conn,
                sub_dir,
                model_name,
                text_data_1,
                text_data_2,
                return_rank,
                label_score.label,
                label_score.score,
                error_msg)
            )

    elif return_rank == "HIGHEST":
        model_output = [
            make_output_row_text_pair(
                bucketfs_conn=bucketfs_conn,
                sub_dir=sub_dir,
                model_name=model_name,
                text_data1=text_data_1,
                text_data2=text_data_2,
                return_rank=return_rank,
                label=label_scores.label_scores[3].label,  # todo what do if not default input?
                score=label_scores.label_scores[3].score,
                error_msg=error_msg)
        ]
    return model_output

def make_number_of_strings(input_str: str, desired_number: int):
    """
    Returns desired number of "input_strX", where X is counting up to desired_number.
    """
    return (input_str + f"{i}" for i in range(desired_number))#todo move these to utils?
