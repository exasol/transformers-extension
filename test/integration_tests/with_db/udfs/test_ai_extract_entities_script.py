from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_for_result_set,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results_multiple_results_per_input,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql


def test_ai_extract_entities_script_without_spans(
    setup_database, db_conn, upload_default_token_classification_model_to_bucketfs
):
    n_rows = 100
    text_data = "The database software company Exasol is based in Nuremberg"
    input_data = []
    for _ in range(n_rows):
        input_data.append((text_data,))

    query = (
        f"SELECT AI_EXTRACT_ENTITIES("
        f"t.text_data"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(text_data));"
    )

    result = db_conn.execute(query).fetchall()

    assert result[0][-1] is None
    # added_columns: start_pos,end_pos,word,entity,score,error_message
    assert_correct_number_of_results_multiple_results_per_input(
        6, 0, input_data[0], result, n_rows
    )

    acceptable_result_sets = [["Exasol", "ORG"], ["Nuremberg", "LOC"]]
    assert_lenient_check_of_output_quality_for_result_set(
        result, acceptable_result_sets, acceptance_factor=0.5, label_index=3
    )
