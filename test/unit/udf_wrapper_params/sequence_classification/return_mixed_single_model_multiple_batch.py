from pathlib import PurePosixPath

from test.unit.udf_wrapper_params.sequence_classification.make_data_row_functions import bucketfs_conn, \
    make_input_row_single_text, make_input_row_text_pair, \
    make_udf_output_for_one_input_row_single_text, make_model_output_for_one_input_row, \
    make_udf_output_for_one_input_row_text_pair

from exasol_udf_mock_python.connection import Connection


class ReturnMixedMultipleModelMultipleBatchComplete:
    """
    multiple model, multiple batch, last batch complete
    """
    #todo desc
    expected_single_text_model_counter = 1
    expected_text_pair_model_counter = 1
    batch_size = 4
    data_size = 2


    tmpdir_name = "_".join(("/tmpdir", __qualname__))
    base_cache_dir = PurePosixPath(tmpdir_name, bucketfs_conn)
    bfs_connections = {
        bucketfs_conn: Connection(address=f"file://{base_cache_dir}")
    }


    inputs_single_text = (
        make_input_row_single_text(return_rank="ALL") * data_size
        + make_input_row_single_text(return_rank="HIGHEST") * data_size
        + make_input_row_single_text(return_rank="HIGHEST") * data_size
        + make_input_row_single_text(return_rank="ALL") * data_size
    )

    output_single_text_row_1 = make_udf_output_for_one_input_row_single_text(
        return_rank="ALL"
    )
    output_single_text_row_2 = make_udf_output_for_one_input_row_single_text(
        return_rank="HIGHEST"
    )
    output_single_text_row_3 = make_udf_output_for_one_input_row_single_text(
        return_rank="HIGHEST"
    )
    output_single_text_row_4 = make_udf_output_for_one_input_row_single_text(
        return_rank="ALL"
    )

    outputs_single_text_batch_1 = output_single_text_row_1 * data_size + output_single_text_row_2 * data_size
    outputs_single_text_batch_2 = output_single_text_row_3 * data_size + output_single_text_row_4 * data_size
    outputs_single_text = outputs_single_text_batch_1 + outputs_single_text_batch_2

    model_output_single_text_one_batch = [make_model_output_for_one_input_row() * batch_size]
    sequence_models_output_df_single_text = [model_output_single_text_one_batch + model_output_single_text_one_batch]

    # ----------------------------------------------------------------
    #todo
    inputs_pair_text = (
        make_input_row_text_pair(return_rank="ALL") * data_size
        + make_input_row_text_pair(return_rank="HIGHEST") * data_size
        + make_input_row_text_pair(return_rank="HIGHEST") * data_size
        + make_input_row_text_pair(return_rank="ALL") * data_size

    )

    output_text_pair_row_1 = make_udf_output_for_one_input_row_text_pair(
        return_rank="ALL"
    )
    output_text_pair_row_2 = make_udf_output_for_one_input_row_text_pair(
        return_rank="HIGHEST"
    )
    output_text_pair_row_3 = make_udf_output_for_one_input_row_text_pair(
        return_rank="HIGHEST"
    )
    output_text_pair_row_4 = make_udf_output_for_one_input_row_text_pair(
        return_rank="ALL"
    )

    outputs_text_pair_batch_1 = output_text_pair_row_1 * data_size + output_text_pair_row_2 * data_size
    outputs_text_pair_batch_2 = output_text_pair_row_3 * data_size + output_text_pair_row_4 * data_size
    outputs_text_pair = outputs_text_pair_batch_1 + outputs_text_pair_batch_2

    model_output_text_pair_one_batch = [make_model_output_for_one_input_row() * batch_size]
    sequence_models_output_df_text_pair = [model_output_text_pair_one_batch + model_output_text_pair_one_batch]

