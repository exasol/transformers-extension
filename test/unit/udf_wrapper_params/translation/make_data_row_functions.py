# default values, used for input/output rows if no other params are given
from dataclasses import dataclass
from typing import Union

device_id = None
bucketfs_conn = "bfs_conn"
sub_dir = "sub_dir"
model_name = "model"

text_data = "text"
source_lanuage = "English"
target_language = "German"

translation_text = "text 1 übersetzt"

max_length = 10
error_msg = None

def make_input_row(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    source_language=source_lanuage,
    target_language=target_language,
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
    target_language=target_language,
    max_length=max_length,
    translation_text=translation_text,
    error_msg=error_msg,
):
    """
    Creates the output row for translation_udf as a list,
    using default values for all parameters that are not specified.
    """
    translation_text = translation_text * max_length if not error_msg else translation_text
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


def make_model_output_for_one_input_row(translation_text=translation_text, max_length=max_length):
    """
    Makes the output the model returns to the udf for one input row.
    returns a list with the model output row.
    each model output row is a dictionary.
    """
    if not translation_text :
        return [{"translation_text": translation_text}]
    else:
        return [{"translation_text": translation_text * max_length}]

