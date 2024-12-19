from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.base_udf.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    sub_dir, model_name, input_data, answer, score, make_model_output_for_one_input_row, make_number_of_strings

class MultipleModelMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete
    """
    expected_model_counter = 2
    batch_size = 2
    data_size = 2
    n_entities = 1

    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir, 2)
    model_name1, model_name2 = make_number_of_strings(model_name, 2)
    input_data1, input_data2 = make_number_of_strings(input_data, 2)
    answer1, answer2 = make_number_of_strings(answer, 2)

    input_data = make_input_row(sub_dir=sub_dir1,
                            model_name=model_name1, input_data=input_data1) * data_size + \
                 make_input_row(sub_dir=sub_dir2,
                            model_name=model_name2, input_data=input_data2) * data_size
    output_data = make_output_row(sub_dir=sub_dir1,
                              model_name=model_name1, input_data=input_data1,
                              answer=answer1, score=score) * n_entities * data_size + \
                  make_output_row(sub_dir=sub_dir2,
                              model_name=model_name2, input_data=input_data2,
                              answer=answer2, score=score + 0.1) * n_entities * data_size


    work_with_span_input_data = make_input_row_with_span(sub_dir=sub_dir1,
                                                         model_name=model_name1, input_data=input_data1) * data_size + \
                                make_input_row_with_span(sub_dir=sub_dir2,
                                                         model_name=model_name2, input_data=input_data2) * data_size
    work_with_span_output_data = make_output_row_with_span(sub_dir=sub_dir1,
                                                           model_name=model_name1, input_data=input_data1,
                                                           answer=answer1,
                                                           score=score) * n_entities * data_size + \
                                 make_output_row_with_span(sub_dir=sub_dir2,
                                                           model_name=model_name2, input_data=input_data2,
                                                           answer=answer2,
                                                           score=score+0.1) * n_entities * data_size

    tokenizer_model_output_df_model1 = [make_model_output_for_one_input_row(number_entities=n_entities, answer=answer1,
                                                                     score=score) * \
                                                                     data_size]
    tokenizer_model_output_df_model2 = [make_model_output_for_one_input_row(number_entities=n_entities, answer=answer2,
                                                                      score=score+0.1) * data_size]

    tokenizer_models_output_df = [tokenizer_model_output_df_model1, tokenizer_model_output_df_model2]


    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }

