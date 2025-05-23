# default values, used for input/output rows if no other params are given
device_id = None  # todo rename variables
bucketfs_conn = "bfs_conn"
sub_dir = "sub_dir"
model_name = "model"

text_data = "text"
text_doc_id = 1
text_start = 0
text_end = 6

agg_strategy_simple = "simple"

token = "token"
token_start = 2
token_end = 4
entity_type = "ENTITY_TYPE"
score = 0.1
error_msg = None


def make_input_row(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    aggregation_strategy=agg_strategy_simple,
):
    """
    Creates an input row for token classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    return [
        (device_id, bucketfs_conn, sub_dir, model_name, text_data, aggregation_strategy)
    ]


def make_output_row(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    aggregation_strategy=agg_strategy_simple,
    start=token_start,
    end=token_end,
    word=token,
    entity=entity_type,
    score=score,
    error_msg=error_msg,
):
    """
    Creates an output row for token classification without span usage as a list,
    using default values for all parameters that are not specified.
    The found token is called "word" in our non span udf output,
    while the type/class of the found token is called "entity".
    """
    return [
        (
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data,
            aggregation_strategy,
            start,
            end,
            word,
            entity,
            score,
            error_msg,
        )
    ]


def make_model_output_for_one_input_row(
    number_entities: int,
    aggregation_strategy=agg_strategy_simple,
    entity_group=entity_type,
    score=score,
    word=token,
    start=token_start,
    end=token_end,
):
    """
    Makes the output the model returns to the udf for one input row.
    The found token is called "word" in the model output,
    while the type/class of the found token is called "entity_group".
    Unless aggregation_strategy == "none", then the type/class of the found
    token is called "entity" in the model output.
    returns a list of number_entities times the model output row.
    each model output row is a dictionary.
    """
    if aggregation_strategy == "none":
        model_output_single_entities = {
            "entity": entity_group,
            "score": score,
            "word": word,
            "start": start,
            "end": end,
        }
    else:
        model_output_single_entities = {
            "entity_group": entity_group,
            "score": score,
            "word": word,
            "start": start,
            "end": end,
        }
    return [[model_output_single_entities] * number_entities]


def make_input_row_with_span(
    device_id=device_id,
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data=text_data,
    text_data_doc_id=text_doc_id,
    text_data_char_begin=text_start,
    text_data_char_end=text_end,
    aggregation_strategy=agg_strategy_simple,
):
    """
    Creates an input row for token classification with span usage as a list,
    using base params for all params that are not specified.
    """
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
            aggregation_strategy,
        )
    ]


def make_output_row_with_span(
    bucketfs_conn=bucketfs_conn,
    sub_dir=sub_dir,
    model_name=model_name,
    text_data_doc_id=text_doc_id,
    text_data_char_begin=text_start,
    text_data_char_end=text_end,
    aggregation_strategy=agg_strategy_simple,
    text_start=text_start,
    token_start=token_start,
    token_end=token_end,
    entity_covered_text=token,
    entity_type=entity_type,
    score=score,
    error_msg=error_msg,
):
    """
    Creates an output row for token classification with span usage as a list,
    using base params for all params that are not specified.
    The found token is called "entity_covered_text" in our non span udf output.
    """
    entity_char_begin = text_start + token_start
    entity_char_end = text_start + token_end
    entity_doc_id = text_data_doc_id
    return [
        (
            bucketfs_conn,
            sub_dir,
            model_name,
            text_data_doc_id,
            text_data_char_begin,
            text_data_char_end,
            aggregation_strategy,
            entity_covered_text,
            entity_type,
            score,
            entity_doc_id,
            entity_char_begin,
            entity_char_end,
            error_msg,
        )
    ]


def make_number_of_strings(input_str: str, desired_number: int):
    """
    Returns desired number of "input_strX", where X is counting up to desired_number.
    """
    return (input_str + f"{i}" for i in range(desired_number))
