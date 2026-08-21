from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql


def run_ai_translate_script_test(n_rows: int, db_conn):
    src_lang = "English"
    target_lang = "German"
    input_data = []
    for _ in range(n_rows):
        input_data.append(
            (
                "The database software company Exasol is based in Nuremberg",
                src_lang,
                target_lang,
            )
        )

    query = (
        f"SELECT AI_TRANSLATE("
        f"t.text_data, "
        f"t.source_language, "
        f"t.target_language"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(text_data, source_language, target_language));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()
    return result, input_data


def test_ai_translate_script(
    setup_database, db_conn, upload_default_translation_model_to_bucketfs
):
    n_rows = 100
    result, input_data = run_ai_translate_script_test(
        n_rows,
        db_conn,
    )

    # assertions
    assert result[0][-1] is None
    added_columns = 2  # translation_text,error_message
    removed_columns = 0
    assert_correct_number_of_results(
        added_columns, removed_columns, input_data[0], result, n_rows
    )

    acceptable_results = ["Die Datenbanksoftware Exasol hat ihren Sitz in Nürnberg"]
    assert_lenient_check_of_output_quality(
        result, acceptable_results, acceptance_factor=0.5, label_index=3
    )
