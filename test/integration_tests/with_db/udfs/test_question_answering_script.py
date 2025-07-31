from test.integration_tests.utils.model_output_quality_checkers import (
    assert_lenient_check_of_output_quality,
)
from test.integration_tests.utils.model_output_result_number_checker import (
    assert_correct_number_of_results,
)
from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params


def test_question_answering_script(
    setup_database, db_conn, upload_question_answering_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    question = "Where is Exasol based?"

    n_rows = 100
    top_k = 1
    input_data = []
    for i in range(n_rows):
        input_data.append(
            (
                "",
                bucketfs_conn_name,
                str(model_params.sub_dir),
                model_params.q_a_model_specs.model_name,
                question,
                "The database software company Exasol is based in Nuremberg",
                top_k,
            )
        )

    query = (
        f"SELECT TE_QUESTION_ANSWERING_UDF("
        f"t.device_id, "
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f"t.model_name, "
        f"t.question, "
        f"t.context_text, "
        f"t.top_k"
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(device_id, bucketfs_conn_name, sub_dir, "
        f"model_name, question, context_text, top_k));"
    )

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None

    # added_columns = answer,score,rank,error_message
    # removed_columns = device_id
    assert_correct_number_of_results(4, 1, input_data[0], result, n_rows)

    acceptable_results = ["Nuremberg", "Germany"]
    assert_lenient_check_of_output_quality(result, top_k, acceptable_results, 2, 6)
