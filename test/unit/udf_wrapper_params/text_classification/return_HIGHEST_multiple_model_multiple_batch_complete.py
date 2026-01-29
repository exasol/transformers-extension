from pathlib import PurePosixPath
from test.unit.udf_wrapper_params.text_classification.make_data_row_functions import (
    bucketfs_conn,
    make_input_row_single_text,
    make_input_row_text_pair,
    make_model_output_for_one_input_row,
    make_udf_output_for_one_input_row_single_text,
    make_udf_output_for_one_input_row_text_pair,
    model_name,
    sub_dir,
)
from test.unit.utils.utils_for_udf_tests import make_number_of_strings

from exasol_udf_mock_python.connection import Connection


class ReturnHighestMultipleModelMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete, return_ranks HIGHEST
    """

    expected_single_text_model_counter = 2
    expected_text_pair_model_counter = 2
    batch_size = 2
    data_size = 2

    return_ranks = "HIGHEST"

    bfs_conn1, bfs_conn2 = make_number_of_strings(bucketfs_conn, 2)
    subdir1, subdir2 = make_number_of_strings(sub_dir, 2)
    model1, model2 = make_number_of_strings(model_name, 2)

    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir1 = PurePosixPath(tmpdir_name, bfs_conn1)
    base_cache_dir2 = PurePosixPath(tmpdir_name, bfs_conn2)
    bfs_connections = {
        bfs_conn1: Connection(address=f"file://{base_cache_dir1}"),
        bfs_conn2: Connection(address=f"file://{base_cache_dir2}"),
    }

    inputs_single_text = (
        make_input_row_single_text(
            bucketfs_conn=bfs_conn1,
            sub_dir=subdir1,
            model_name=model1,
            return_ranks=return_ranks,
        )
        * data_size
        + make_input_row_single_text(
            bucketfs_conn=bfs_conn2,
            sub_dir=subdir2,
            model_name=model2,
            return_ranks=return_ranks,
        )
        * data_size
    )

    output_single_text_model_1 = make_udf_output_for_one_input_row_single_text(
        bucketfs_conn=bfs_conn1,
        sub_dir=subdir1,
        model_name=model1,
        return_ranks=return_ranks,
    )
    output_single_text_model_2 = make_udf_output_for_one_input_row_single_text(
        bucketfs_conn=bfs_conn2,
        sub_dir=subdir2,
        model_name=model2,
        return_ranks=return_ranks,
    )

    outputs_single_text = (
        output_single_text_model_1 * data_size + output_single_text_model_2 * data_size
    )

    model_output_single_text_model_1 = [
        make_model_output_for_one_input_row() * data_size
    ]
    model_output_single_text_model_2 = [
        make_model_output_for_one_input_row() * data_size
    ]
    text_class_models_output_df_single_text = [
        model_output_single_text_model_1,
        model_output_single_text_model_2,
    ]

    # ----------------------------------------------------------------

    inputs_pair_text = (
        make_input_row_text_pair(
            bucketfs_conn=bfs_conn1,
            sub_dir=subdir1,
            model_name=model1,
            return_ranks=return_ranks,
        )
        * data_size
        + make_input_row_text_pair(
            bucketfs_conn=bfs_conn2,
            sub_dir=subdir2,
            model_name=model2,
            return_ranks=return_ranks,
        )
        * data_size
    )

    output_text_pair_1 = make_udf_output_for_one_input_row_text_pair(
        bucketfs_conn=bfs_conn1,
        sub_dir=subdir1,
        model_name=model1,
        return_ranks=return_ranks,
    )
    output_text_pair_2 = make_udf_output_for_one_input_row_text_pair(
        bucketfs_conn=bfs_conn2,
        sub_dir=subdir2,
        model_name=model2,
        return_ranks=return_ranks,
    )

    outputs_text_pair = output_text_pair_1 * data_size + output_text_pair_2 * data_size
    text_class_models_output_df_text_pair = [
        model_output_single_text_model_1,
        model_output_single_text_model_2,
    ]
