from tests.utils.parameters import model_params


def test_named_entity_recognition_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    n_rows = 100
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.name,
            model_params.text_data))

    query = f"SELECT TE_NAMED_ENTITY_RECOGNITION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(device_id, bucketfs_conn_name, " \
            f"sub_dir, model_name, text_data));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()
    print(result)

    # assertions
    n_cols_result = len(input_data[0]) + 2  # + 2 new cols -1 device_id col
    assert len(result) >= n_rows and len(result[0]) == n_cols_result
