from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params

def assert_correct_number_of_results(
    added_columns: int,
    removed_columns: int,
    input_data_row: tuple,
    result: list,
    n_rows: int,
):
    n_cols_result = len(input_data_row) + (added_columns - removed_columns)
    assert len(result) >= n_rows and len(result[0]) == n_cols_result, (
        f"format of result is not correct,"
        f"expected >= {n_rows} rows, {n_cols_result} columns."
        f"actual: {len(result)} rows, {len(result[0])} columns"
    )


def test_list_models_script(
    setup_database, db_conn, upload_token_classification_model_to_bucketfs, upload_translation_model_to_bucketfs,
    upload_filling_mask_model_to_bucketfs
):
    bucketfs_conn_name, _ = setup_database
    input_data = [["", bucketfs_conn_name, str(model_params.sub_dir)]]# todo do one with actuall subdir?
    expected_result = [] #todo
    input_data_subdir_not_exist = ["non-existend-subdir", bucketfs_conn_name, str(model_params.sub_dir)]
    input_data_subdir_empty = []

    query = (
        f"SELECT TE_LS_MODELS_UDF("
        f"t.bucketfs_conn_name, "
        f"t.sub_dir, "
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(bucketfs_conn_name, "
        f"sub_dir));"
    )

    # execute UDF
    result = db_conn.execute(query).fetchall()
    # assertions
    assert result[0][-1] is None
    # added_columns: model_name, version, task_name, seed, path, error_message
    assert_correct_number_of_results(6, 0, input_data, result, n_rows)
    # todo assert output correct
