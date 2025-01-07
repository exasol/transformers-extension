# default values, used for input/output rows if no other params are given
device_id=None
bucketfs_conn="bfs_conn"
sub_dir="sub_dir"
model_name="model"

text_data="text"
candidate_labels="simple"
label="label"
rank=1

score=0.1
error_msg = None

def make_input_row(device_id=device_id, bucketfs_conn=bucketfs_conn, sub_dir=sub_dir,
                   model_name=model_name, text_data=text_data, candidate_labels=candidate_labels):
    """
    Creates an input row for zero shot classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    return [(device_id, bucketfs_conn, sub_dir, model_name,text_data, candidate_labels)]

def make_output_row(bucketfs_conn=bucketfs_conn, sub_dir=sub_dir,
                    model_name=model_name, text_data=text_data, candidate_labels=candidate_labels,
                    label=label, score=score, rank=rank, error_msg=error_msg):
    """
    Creates an output row for zero shot classification without span usage as a list,
    using default values for all parameters that are not specified.
    """
    return [(bucketfs_conn, sub_dir, model_name,text_data, candidate_labels,
             label,score, rank, error_msg)]

def make_model_output_for_one_input_row(candidate_labels=candidate_labels, score=score):
    """
    Makes the output the model returns to the udf for one input row.
    returns a list with the model output row.
    each model output row is a dictionary.
    """
    model_output = []
    for c_label in candidate_labels.split(","):
        model_output_single_label = {'labels': c_label, 'scores': score}
        score += 0.1
        model_output.append(model_output_single_label)
    return [model_output]


def make_number_of_strings(input_str: str, desired_number: int):
    """
    Returns desired number of "input_strX", where X is counting up to desired_number.
    """
    return (input_str + f"{i}" for i in range(desired_number))

