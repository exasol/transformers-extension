# default values, used for input/output rows if no other params are given
from dataclasses import dataclass
from typing import Union

device_id = None
bucketfs_conn = "bfs_conn"
sub_dir = "sub_dir"
model_name = "model"

text_data = "text"
source_lanuage = "English"
target_language_1 = "German"
target_language_2 = "French"

translation_text_1 = "text 1 übersetzt"
translation_text_2 = "text 1 traduit"

max_length = 10 #todo does this one work?
error_msg = None

def make_input_row(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    source_language=source_lanuage,
    target_language=target_language_1,
    max_length=max_length
):
    """
    Creates an input row for translation udf as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (
            device_id,
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            source_language,
            target_language,
            max_length
        )
    ]


def make_udf_output_for_one_input_row(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    source_language=source_lanuage,
    target_language=target_language_1,
    max_length=max_length,
    translation_text=translation_text_1,
    error_msg=error_msg,
):
    """
    Creates the output row for translation_udf as a list,
    using default values for all parameters that are not specified.
    """

    return [
        (
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            source_language,
            target_language,
            max_length,
            translation_text,
            error_msg
        )
    ]


def make_model_output_for_one_input_row(target_language=target_language_1, max_len=max_length):
    """
    Makes the output the model returns to the udf for one input row.
    returns a list with the model output row.
    each model output row is a dictionary.
    """
    if target_language == target_language_1:
        return [{"translation_text": translation_text_1 * max_len}]
    elif target_language == target_language_2:
        return [{"translation_text": translation_text_2 * max_len}]
    else: return [{"translation_text": None}]


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
