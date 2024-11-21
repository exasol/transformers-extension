from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    token, score, make_model_output_for_one_input_row, make_number_of_strings, model_name, sub_dir, text_data


class MultipleModelMultipleBatchMultipleModelsPerBatch:
    """
    multiple model, multiple batch, multiple models per batch
    """
    expected_model_counter = 4
    batch_size = 2
    data_size = 1
    n_entities = 3

    bfs_conn1, bfs_conn2, bfs_conn3, bfs_conn4 = make_number_of_strings(bucketfs_conn, 4)
    subdir1, subdir2, subdir3, subdir4 = make_number_of_strings(sub_dir, 4)
    model_name1, model_name2, model_name3, model_name4 = make_number_of_strings(model_name, 4)
    text1, text2, text3, text4 = make_number_of_strings(text_data, 4)
    token1, token2, token3, token4 = make_number_of_strings(token, 4)

    input_data = make_input_row(bucketfs_conn=bfs_conn1, sub_dir=subdir1,
                                model_name=model_name1, text_data=text1) * data_size + \
                 make_input_row(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                               model_name=model_name2, text_data=text2) * data_size + \
                 make_input_row(bucketfs_conn=bfs_conn3, sub_dir=subdir3,
                                model_name=model_name3, text_data=text3) * data_size + \
                 make_input_row(bucketfs_conn=bfs_conn4, sub_dir=subdir4,
                                model_name=model_name4, text_data=text4) * data_size
    output_data = make_output_row(bucketfs_conn=bfs_conn1, sub_dir=subdir1,
                                  model_name=model_name1, text_data=text1,
                                  word=token1, score=score) * n_entities * data_size + \
                  make_output_row(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                                  model_name=model_name2, text_data=text2,
                                  word=token2, score=score+0.1) * n_entities * data_size + \
                  make_output_row(bucketfs_conn=bfs_conn3, sub_dir=subdir3,
                                  model_name=model_name3, text_data=text3,
                                  word=token3, score=score+0.2) * n_entities * data_size + \
                  make_output_row(bucketfs_conn=bfs_conn4, sub_dir=subdir4,
                                  model_name=model_name4, text_data=text4,
                                  word=token4, score=score+0.3) * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span(bucketfs_conn=bfs_conn1, sub_dir=subdir1,
                                                         model_name=model_name1, text_data=text1) * data_size + \
                                make_input_row_with_span(bucketfs_conn=bfs_conn2, sub_dir=subdir2,
                                                          model_name=model_name2, text_data=text2) * data_size + \
                                make_input_row_with_span(bucketfs_conn=bfs_conn3, sub_dir=subdir3,
                                                          model_name=model_name3, text_data=text3) * data_size + \
                                make_input_row_with_span(bucketfs_conn=bfs_conn4, sub_dir=subdir4,
                                                          model_name=model_name4, text_data=text4) * data_size
    work_with_span_output_data = make_output_row_with_span(bucketfs_conn=bfs_conn1, sub_dir=subdir1, model_name=model_name1,
                                                           entity_covered_text=token1, score=score) * n_entities * data_size + \
                                 make_output_row_with_span(bucketfs_conn=bfs_conn2, sub_dir=subdir2,model_name=model_name2,
                                                           entity_covered_text=token2, score=score+0.1) * n_entities * data_size + \
                                 make_output_row_with_span(bucketfs_conn=bfs_conn3, sub_dir=subdir3,model_name=model_name3,
                                                           entity_covered_text=token3, score=score+0.2) * n_entities * data_size + \
                                 make_output_row_with_span(bucketfs_conn=bfs_conn4, sub_dir=subdir4,model_name=model_name4,
                                                           entity_covered_text=token4, score=score+0.3) * n_entities * data_size

    tokenizer_model_output_df_model1 =  [make_model_output_for_one_input_row(word=token1, score=score, number_entities=n_entities) * data_size]
    tokenizer_model_output_df_model2 =  [make_model_output_for_one_input_row(word=token2, score=score+0.1, number_entities=n_entities) * data_size]
    tokenizer_model_output_df_model3 =  [make_model_output_for_one_input_row(word=token3, score=score+0.2, number_entities=n_entities) * data_size]
    tokenizer_model_output_df_model4 =  [make_model_output_for_one_input_row(word=token4, score=score+0.3, number_entities=n_entities) * data_size]


    tokenizer_models_output_df = [tokenizer_model_output_df_model1, tokenizer_model_output_df_model2,
                                  tokenizer_model_output_df_model3, tokenizer_model_output_df_model4]



    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    base_cache_dir3 = PurePosixPath(tmpdir_name, bfs_conn3)
    base_cache_dir4 = PurePosixPath(tmpdir_name, bfs_conn4)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}"),
        bfs_conn3: Connection(address=f"file://{base_cache_dir3}"),
        bfs_conn4: Connection(address=f"file://{base_cache_dir4}")
    }
