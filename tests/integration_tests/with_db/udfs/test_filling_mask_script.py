from typing import Tuple, Any, List

from tests.utils.parameters import model_params


def python_value_to_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"Type {type(value)} of value {value} not supported.")


def python_row_to_sql(row: Tuple) -> str:
    sql_values = [python_value_to_sql(value) for value in row]
    sql_row_str = ",".join(sql_values)
    return f"({sql_row_str})"


def python_rows_to_sql(rows: List[Tuple]) -> str:
    sql_rows = [python_row_to_sql(row) for row in rows]
    sql_rows_str = ",".join(sql_rows)
    return sql_rows_str


def test_filling_mask_script(
        setup_database, pyexasol_connection, upload_base_model_to_bucketfs):
    bucketfs_conn_name, schema_name = setup_database
    text_data = "Exasol is an analytics <mask> management software company."
    n_rows = 100
    top_k = 3
    input_data = []
    for i in range(n_rows):
        input_data.append((
            '',
            bucketfs_conn_name,
            None,
            str(model_params.sub_dir),
            model_params.base_model,
            text_data,
            top_k))

    query = f"SELECT TE_FILLING_MASK_UDF(" \
            f"t.device_id, " \
            f"t.bucketfs_conn_name, " \
            f"t.token_conn_name, " \
            f"t.sub_dir, " \
            f"t.model_name, " \
            f"t.text_data," \
            f"t.top_k" \
            f") FROM (VALUES {python_rows_to_sql(input_data)} " \
            f"AS t(device_id, bucketfs_conn_name, token_conn_name, sub_dir, " \
            f"model_name, text_data, top_k));"

    # execute sequence classification UDF
    result = pyexasol_connection.execute(query).fetchall()

    # assertions
    added_columns = 4  # filled_text,score,rank,error_message
    removed_columns = 1  # device_id col
    n_rows_result = n_rows * top_k
    n_cols_result = len(input_data[0]) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result
