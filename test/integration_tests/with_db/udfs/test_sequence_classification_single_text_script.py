from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality_with_score,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_sequence_classification_single_text_script(
    setup_database, db_conn, upload_sequence_classification_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    n_labels = 3  # negative, neutral, positive

    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.sequence_class_model_specs.model_name,
                "I am so happy to be working on the Transformers Extension.",
            )
        )

    query = (
        f"SELECT TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.text_data) "
        f"FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, "
        f"sub_dir, model_name, text_data));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    n_rows_result = n_rows * n_labels
    # added_columns: label,score,error_message
    # removed_columns: device_id,
    assert_correct_number_of_results(3, 1, input_data[0], result, n_rows_result)

    # Since in this test the input is a sentence with positive sentiment, which the test model can detect,
    # the "acceptable_results" here is the label "positive" with a reasonably high score.
    acceptable_results = ["positive"]
    assert_lenient_check_of_output_quality_with_score(
        result, n_rows_result, acceptable_results, 1.5, label_index=4
    )
