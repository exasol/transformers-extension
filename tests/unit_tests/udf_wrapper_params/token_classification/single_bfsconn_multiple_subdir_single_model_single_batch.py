from pathlib import PurePosixPath
from exasol_udf_mock_python.connection import Connection
from tests.unit_tests.udf_wrapper_params.token_classification.make_data_row_functions import make_input_row, \
    make_output_row, make_input_row_with_span, make_output_row_with_span, bucketfs_conn, \
    sub_dir, token, score, make_model_output_for_one_input_row, make_number_of_strings


class SingleBucketFSConnMultipleSubdirSingleModelNameSingleBatch:
    """
    single bucketfs connection, multiple subdir, single model, single batch
    """
    expected_model_counter = 2
    batch_size = 4
    data_size = 2
    n_entities = 3

    sub_dir1, sub_dir2 = make_number_of_strings(sub_dir, 2)
    token1, token2 = make_number_of_strings(token, 2)

    input_data = make_input_row(sub_dir=sub_dir1) * data_size + \
                 make_input_row(sub_dir=sub_dir2) * data_size
    output_data = make_output_row(sub_dir=sub_dir1, word=token1, score=score) * n_entities * data_size + \
                  make_output_row(sub_dir=sub_dir2, word=token2, score=score+0.1) * n_entities * data_size

    work_with_span_input_data = make_input_row_with_span(sub_dir=sub_dir1) * data_size + \
                                make_input_row_with_span(sub_dir=sub_dir2) * data_size
    work_with_span_output_data =  make_output_row_with_span(sub_dir=sub_dir1, entity_covered_text=token1, score=score) * n_entities * data_size + \
                                  make_output_row_with_span(sub_dir=sub_dir2, entity_covered_text=token2, score=score+0.1) * n_entities * data_size

    tokenizer_model_output_df = [make_model_output_for_one_input_row(word=token1, score=score, number_entities=n_entities) * data_size] + \
                                [make_model_output_for_one_input_row(word=token2, score=score, number_entities=n_entities) * data_size]

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir1}")
    }

