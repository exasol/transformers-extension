# default values, used for input/output rows if no other params are given
device_id=None
bucketfs_conn="bfs_conn"
sub_dir="sub_dir"
model_name="model"

input_data="input_text"

test_span_column_drop="drop_this"
test_span_column_add="add_this"

answer="answer"
score=0.1
error_msg = None

def make_input_row(device_id=device_id, model_name=model_name, sub_dir=sub_dir,
                   bucketfs_conn=bucketfs_conn, input_data=input_data):
    """
    Creates an input row for answer classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    return [(device_id, model_name, sub_dir, bucketfs_conn, input_data)]

def make_output_row(model_name=model_name, sub_dir=sub_dir,
                   bucketfs_conn=bucketfs_conn, input_data=input_data,
                    answer=answer, score=score, error_msg=error_msg):
    """
    Creates an output row for answer classification without span usage as a list,
    using default values for all parameters that are not specified.
    The found answer is called "answer" in our non span udf output
    """
    return [( model_name, sub_dir, bucketfs_conn, input_data, answer, score, error_msg)]

def make_model_output_for_one_input_row(number_entities:int, answer=answer, score=score):
    """
    Makes the output the model returns to the udf for one input row.
    The found answer is called "answer" in the model output.
    Unless aggregation_strategy == "none", then the type/class of the found.
    returns a list of number_entities times the model output row.
    each model output row is a dictionary.
    """
    model_output_single_entities = {'answer': answer, 'score': score}

    return [[model_output_single_entities] * number_entities] #todo test where this is not in list?


def make_input_row_with_span(device_id=device_id, bucketfs_conn=bucketfs_conn, sub_dir=sub_dir,
                             model_name=model_name, input_data=input_data,
                             test_span_column_drop=test_span_column_drop):
    """
    Creates an input row for answer classification with span usage as a list,
    using base params for all params that are not specified.
    """
    return [(device_id, model_name, sub_dir, bucketfs_conn, input_data, test_span_column_drop)]

def make_output_row_with_span(bucketfs_conn=bucketfs_conn, sub_dir=sub_dir,
                              model_name=model_name, input_data=input_data,
                              test_span_column_add=test_span_column_add,
                              answer=answer, score=score, error_msg=error_msg):
    """
    Creates an output row for answer classification with span usage as a list,
    using base params for all params that are not specified.
    The found answer is called "answer" in our non span udf output.#todo update docstrings
    """
    return [( model_name, sub_dir, bucketfs_conn, input_data,
              answer, score, test_span_column_add, error_msg)]


def make_number_of_strings(input_str: str, desired_number: int):#todo move this to utils?
    """
    Returns desired number of "input_strX", where X is counting up to desired_number.
    """
    return (input_str + f"{i}" for i in range(desired_number))
