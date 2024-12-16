# default values, used for input/output rows if no other params are given
device_id=None
bucketfs_conn="bfs_conn"
sub_dir="sub_dir"
model_name="model"
token_conn=None

answer="answer"
score=0.1
error_msg = None

def make_input_row(device_id=device_id, model_name=model_name, sub_dir=sub_dir,
                   bucketfs_conn=bucketfs_conn, token_conn=token_conn):
    """
    Creates an input row for token classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    return [(device_id, model_name, sub_dir, bucketfs_conn, token_conn)]

def make_output_row(model_name=model_name, sub_dir=sub_dir,
                   bucketfs_conn=bucketfs_conn, token_conn=token_conn,
                    answer=answer, score=score, error_msg=error_msg):
    """
    Creates an output row for token classification without span usage as a list,
    using default values for all parameters that are not specified.
    The found token is called "word" in our non span udf output,
    while the type/class of the found token is called "entity".
    """
    return [( model_name, sub_dir, bucketfs_conn, token_conn, answer, score, error_msg)]

def make_model_output_for_one_input_row(number_entities:int, answer=answer, score=score):
    """
    Makes the output the model returns to the udf for one input row.
    The found token is called "word" in the model output,
    while the type/class of the found token is called "entity_group".
    Unless aggregation_strategy == "none", then the type/class of the found
    token is called "entity" in the model output.
    returns a list of number_entities times the model output row.
    each model output row is a dictionary.
    """
    model_output_single_entities = {'answer': answer, 'score': score}

    return [[model_output_single_entities] * number_entities] #todo test where this is not in list?


def make_number_of_strings(input_str: str, desired_number: int):#todo move this to utils?
    """
    Returns desired number of "input_strX", where X is counting up to desired_number.
    """
    return (input_str + f"{i}" for i in range(desired_number))

