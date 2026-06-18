from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_with_score,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql


def test_ai_classify_script(
    setup_database,
    db_conn,
    upload_default_classify_model_to_bucketfs,
):
    n_rows = 100
    candidate_labels = "Database,Analytics,Germany,Food,Party"
    text_data = "The database software company Exasol is based in Nuremberg"

    input_data = []
    for _ in range(n_rows):
        input_data.append(
            (
                text_data,
                candidate_labels,
            )
        )

    query = (
        f"SELECT AI_CLASSIFY("
        f"t.text_data, "
        f"t.candidate_labels) "
        f"FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(text_data, candidate_labels));"
    )

    # execute UDF
    result = db_conn.execute(query).fetchall()

    assert result[0][-1] is None
    # added_columns: label,score,error_message
    assert_correct_number_of_results(3, 0, input_data[0], result, n_rows)

    acceptable_results = ["Analytics", "Database", "Germany"]
    assert_lenient_check_of_output_quality_with_score(
        result, acceptable_results, 1 / 1.8, label_index=2
    )
