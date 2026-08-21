from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql


def test_ai_answer_script(
    setup_database, db_conn, upload_default_question_answering_model_to_bucketfs
):
    question = "Where is Exasol based?"

    n_rows = 100
    input_data = []
    for _ in range(n_rows):
        input_data.append(
            (
                question,
                "The database software company Exasol is based in Nuremberg",
            )
        )

    query = (
        f"SELECT AI_ANSWER("
        f"t.question, "
        f"t.context_text"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(question, context_text));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    # added_columns : answer,error_message
    assert_correct_number_of_results(2, 0, input_data[0], result, n_rows)

    acceptable_results = [
        "Nuremberg",
        "Germany",
    ]
    assert_lenient_check_of_output_quality(result, acceptable_results, 0.5, 2)
