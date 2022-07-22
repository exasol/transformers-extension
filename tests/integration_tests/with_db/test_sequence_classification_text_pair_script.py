from tests.utils.parameters import model_params, bucketfs_params


def test_sequence_classification_single_text_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    n_labels = 2
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append((
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.name,
            model_params.text_data,
            ' '.join((model_params.text_data, str(i)))))

    query = f"SELECT TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF(" \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.first_text, " \
            f"t.second_text" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(bucketfs_conn_name, sub_dir, model_name, " \
            f"first_text, second_text));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()

    # assertions
    assert len(result) == n_rows * n_labels and \
           len(result[0]) == len(input_data[0]) + 2

