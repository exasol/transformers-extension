from tests.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from tests.utils.parameters import model_params


def test_filling_mask_script(
        setup_database, db_conn, upload_base_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    text_data = "Exasol is an analytics <mask> management software company."
    n_rows = 100
    top_k = 3
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            str(model_params.sub_dir),
            model_params.base_model_specs.model_name,
            text_data,
            top_k))

    query = f"SELECT TE_FILLING_MASK_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data," \
            f"t.top_k" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, sub_dir, " \
            f"model_name, text_data, top_k));"

    # execute sequence classification UDF
    result = db_conn.execute(query).fetchall()

    # assertions
    assert result[0][-1] is None
    added_columns = 4  # filled_text,score,rank,error_message
    removed_columns = 1  # device_id col
    n_rows_result = n_rows * top_k
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result
