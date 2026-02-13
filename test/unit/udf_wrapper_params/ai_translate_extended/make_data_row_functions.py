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

translation_text = "text übersetzt, "

max_new_tokens = 10
error_msg = None


def make_input_row(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    source_language=source_lanuage,
    target_language=target_language,
    max_new_tokens=max_new_tokens,
):
    """
    Creates an input row for ai_translate_extended udf as a list,
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
            max_new_tokens,
        )
    ]


def make_udf_output_for_one_input_row(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    source_language=source_lanuage,
    target_language=target_language,
    max_new_tokens=max_new_tokens,
    translation_text=translation_text,
    error_msg=error_msg,
):
    """
    Creates the output row for translation_udf as a list,
    using default values for all parameters that are not specified.
    """
    translation_text = (
        translation_text + str(max_new_tokens) if not error_msg else translation_text
    )
    return [
        (
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            source_language,
            target_language,
            max_new_tokens,
            translation_text,
            error_msg,
        )
    ]


def translation_models_output_generator(input_texts, max_new_tokens):
    """
    Makes the output the model returns to the udf for one input batch.
    returns a list with the model output.
    each model output row is a dictionary.
    """
    output = []
    for input_text in input_texts:
        # throw an error for "error on pred" test cases
        if "error" in input_text:
            return Exception(
                "Traceback mock_pipeline is throwing an error intentionally"
            )
        # remove the input prefix from the input
        input_prefix, text = input_text.split(": ")
        lang_marker = " übersetzt, "
        if "French" in input_prefix:
            lang_marker = " traduit, "
        else:
            lang_marker = " übersetzt, "
        output.append({"translation_text": (text + lang_marker) + str(max_new_tokens)})
    return output
