from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_text_generation_script(
    setup_database, db_conn, upload_text_generation_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    text_data = "Exasol is an analytics database management"
    n_rows = 100
    max_length = 12
    return_full_text = True
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.text_gen_model_specs.model_name,
                text_data,
                max_length,
                return_full_text,
            )
        )

    query = (
        f"SELECT TE_TEXT_GENERATION_UDF("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data, "
        f"t.max_length,"
        f"t.return_full_text"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, sub_dir, "
        f"model_name, text_data, max_length, return_full_text));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    # added_columns = generated_text,error_message
    # removed_columns = device_id
    assert_correct_number_of_results(2, 1, input_data[0], result, n_rows)

    acceptable_results = ["software", "system", "solution", "tool"]
    assert_lenient_check_of_output_quality(result, acceptable_results, 0.5, 6)
