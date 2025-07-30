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
    added_columns = 4  # answer,score,rank,error_message
    removed_columns = 1  # device_id col
    n_rows_result = n_rows
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

    results = [result[i][6] for i in range(len(result))]

    # Lenient test for quality of results.
    # We do this by seeing if the result contains one of our predefined "acceptable_results".
    # This check is only here to assure us the models output is not totally of kilter
    # (and crucially does not get worse with our changes over time),
    # and therefore we can assume model loading and execution is working correctly.
    # We to make this check deterministic in the future.
    acceptable_results = ["Nuremberg", "Germany"]
    number_accepted_results = 0

    def contains(string, list):
        return any(map(lambda x: x in string, list))

    for i in range(len(results)):
        if contains(results[i], acceptable_results):
            number_accepted_results += 1
    assert number_accepted_results > top_k / 2
