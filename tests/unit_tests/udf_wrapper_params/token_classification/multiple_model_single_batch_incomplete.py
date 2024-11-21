from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    sub_dir, model_name, text_data, token, entity_type, score, make_model_output_for_one_input_row, make_number_of_strings

class MultipleModelSingleBatchIncomplete:
    """
    multiple model, single batch, last batch incomplete
    """
    expected_model_counter = 2
    batch_size = 5
    data_size = 2
    n_entities = 3

    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir, 2)
    model_name1, model_name2 = make_number_of_strings(model_name, 2)
    text_data1, text_data2 = make_number_of_strings(text_data, 2)
    token1, token2 = make_number_of_strings(token, 2)
    entity_type1, entity_type2 = make_number_of_strings(entity_type, 2)


    input_data = make_input_row(sub_dir=sub_dir1,
                            model_name=model_name1, text_data=text_data1) * data_size + \
                 make_input_row(sub_dir=sub_dir2,
                            model_name=model_name2, text_data=text_data2) * data_size
    output_data = make_output_row(sub_dir=sub_dir1,
                              model_name=model_name1, text_data=text_data1,
                              word=token1, entity=entity_type1, score=score) * n_entities * data_size + \
                  make_output_row(sub_dir=sub_dir2,
                              model_name=model_name2, text_data=text_data2,
                              word=token2, entity=entity_type2, score=score + 0.1) * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span(sub_dir=sub_dir1,
                                                         model_name=model_name1, text_data=text_data1) * data_size + \
                                make_input_row_with_span(sub_dir=sub_dir2,
                                                         model_name=model_name2, text_data=text_data2) * data_size
    work_with_span_output_data = make_output_row_with_span(sub_dir=sub_dir1,
                                                           model_name=model_name1, entity_covered_text=token1,
                                                           entity_type=entity_type1, score=score) * n_entities * data_size + \
                                 make_output_row_with_span(sub_dir=sub_dir2,
                                                           model_name=model_name2, entity_covered_text=token2,
                                                           entity_type=entity_type2, score=score+0.1) * n_entities * data_size

    tokenizer_model_output_df_model1 = [make_model_output_for_one_input_row(number_entities=n_entities, word=token1,
                                                                     entity_group=entity_type1, score=score) * \
                                                                     data_size]
    tokenizer_model_output_df_model2 = [make_model_output_for_one_input_row(number_entities=n_entities, word=token2,
                                                                     entity_group=entity_type2, score=score+0.1) * data_size]

    tokenizer_models_output_df = [tokenizer_model_output_df_model1, tokenizer_model_output_df_model2]



    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }

