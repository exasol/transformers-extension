from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_with_score,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql


def test_ai_custom_classify_extended_script(
    setup_database,
    db_conn,
    upload_default_sentiment_model_to_bucketfs,
):
    n_rows = 100

    input_data = [
        ("I am so happy to be working on the Transformers Extension.",) * n_rows
    ]

    query = (
        f"SELECT AI_SENTIMENT_EXTENDED("
        f"t.text_data) "
        f"FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(text_data));"
    )

    # execute UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    n_rows_result = n_rows
    # added_columns: label,score,error_message
    assert_correct_number_of_results(3, 0, input_data[0], result, n_rows_result)

    # Since in this test the input is a sentence with positive sentiment,
    # which the default sentiment model can detect,
    # the "acceptable_results" here is the label "positive" with a reasonably high score.
    acceptable_results = ["positive"]
    assert_lenient_check_of_output_quality_with_score(
        result, acceptable_results, 1 / 1.5, label_index=5
    )
