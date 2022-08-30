from tests.utils.parameters import model_params


def test_text_generation_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_base_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    text_data = "Exasol is an analytics database management"
    n_rows = 100
    max_length = 10
    return_full_text = True
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.base,
            text_data,
            max_length,
            return_full_text
        ))

    query = f"SELECT TE_TEXT_GENERATION_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data, " \
            f"t.max_length," \
            f"t.return_full_text" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
            f"model_name, text_data, max_length, return_full_text));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()
    print(result)

    # assertions
    n_rows_result = n_rows
    n_cols_result = len(input_data[0])   # + 1 new cols -1 device_id col
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result

