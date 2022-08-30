from tests.utils.parameters import model_params


def test_question_answering_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_base_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    question = "Where is the Exasol?"
    n_rows = 100
    top_k = 1
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.base,
            question,
            ' '.join((model_params.text_data, str(i))),
            top_k
        ))

    query = f"SELECT TE_QUESTION_ANSWERING_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.question, " \
            f"t.context_text, " \
            f"t.top_k" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
            f"model_name, question, context_text, top_k));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()
    print(result)

    # assertions
    n_rows_result = n_rows
    n_cols_result = len(input_data[0]) + 1  # + 2 new cols -1 device_id col
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

