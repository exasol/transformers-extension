from tests.utils.parameters import model_params, bucketfs_params


def test_filling_mask_script(
        upload_language_container, setup_database,
        pyexasol_connection, upload_model_to_bucketfs):

    bucketfs_conn_name, schema_name = setup_database
    text_data = "Exasol is an analytics <mask> management software company."
    n_rows = 100
    top_k = 3
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.name,
            text_data,
            top_k))

    query = f"SELECT TE_FILLING_MASK_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data," \
            f"t.top_k" \
            f") FROM (VALUES {str(tuple(input_data))} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
            f"model_name, text_data, top_k));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()
    print(result)

    # assertions
    n_rows_result = n_rows * top_k
    n_cols_result = len(input_data[0]) + 1  # + 2 new cols -1 device_id col
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result
