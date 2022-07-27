from tests.utils.parameters import model_params, bucketfs_params


def test_question_answering_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    question = "Where is the Exasol?"
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append((
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.name,
            question,
            ' '.join((model_params.text_data, str(i)))))

    query = f"SELECT TE_QUESTION_ANSWERING_UDF(" \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.question, " \
            f"t.context_text" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(bucketfs_conn_name, sub_dir, model_name, " \
            f"question, context_text));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()

    # assertions
    assert len(result) == n_rows  and \
           len(result[0]) == len(input_data[0]) + 2

