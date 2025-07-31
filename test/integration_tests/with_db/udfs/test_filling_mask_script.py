from test.integration_tests.utils.model_output_quality_checkers import assert_lenient_check_of_output_quality
from test.integration_tests.utils.model_output_result_number_checker import assert_correct_number_of_results
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_filling_mask_script(
    setup_database, db_conn, upload_filling_mask_model_to_bucketfs
):
    bucketfs_conn_name, schema_name = setup_database
    text_data = "I <mask> you so much."
    n_rows = 100
    top_k = 3
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.base_model_specs.model_name,
                text_data,
                top_k,
            )
        )

    query = (
        f"SELECT TE_FILLING_MASK_UDF("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data,"
        f"t.top_k"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, sub_dir, "
        f"model_name, text_data, top_k));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    n_rows_result = n_rows * top_k
    # added_columns = filled_text,score,rank,error_message
    # removed_columns = device_id
    assert_correct_number_of_results(4, 1, input_data[0],
                                     result, n_rows_result)

    acceptable_results = ["love", "miss", "want", "need"]
    assert_lenient_check_of_output_quality(result, n_rows_result, acceptable_results,2, 5)
