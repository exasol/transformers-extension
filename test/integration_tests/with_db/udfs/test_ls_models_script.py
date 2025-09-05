from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.utils.parameters import model_params
from test.utils.parameters import PATH_IN_BUCKET

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
    setup_database, db_conn, upload_tiny_model_to_bucketfs#, upload_translation_model_to_bucketfs,
    #upload_filling_mask_model_to_bucketfs#todo add fixture which just creates placeholder files instead of using models here?
):
    bucketfs_conn_name, _ = setup_database
    subdir = "model_sub_dir"


    #subdir = str(model_params.sub_dir) # this is not set for most models in our params
    input_data = [(bucketfs_conn_name, subdir)]
    expected_result = [(bucketfs_conn_name, 'model_sub_dir', 'prajjwal1/bert-tiny', 'task', '/buckets/bfsdefault/default/container/model_sub_dir/prajjwal1/bert-tiny_task', None)] #todo
    input_data_subdir_not_exist = [(bucketfs_conn_name, "non-existend-subdir")]
    expected_result_non_existend_subdir = [(bucketfs_conn_name, 'non-existend-subdir', None, None, None, 'no models in this subdir')]#todo do we wat different message
    input_data_subdir_empty_string = [(bucketfs_conn_name, "")]
    expected_result_data_subdir_empty_string = [(bucketfs_conn_name, 'None', None, None, None, 'no models in this subdir')]

    input_data_sets = [input_data, input_data_subdir_not_exist, input_data_subdir_empty_string]

    for input_data_set in input_data_sets:
        query = (
            f"SELECT TE_LS_MODELS_UDF("
            f"t.bucketfs_conn_name, "
            f"t.sub_dir "
            f") FROM (VALUES {python_rows_to_sql(input_data_set)} "
            f"AS t(bucketfs_conn_name, "
            f"sub_dir));"
        )

        # execute UDF
        result = db_conn.execute(query).fetchall()
        for item in result:
            print(item)
        # added_columns: model_name, task_name, path, error_message
        assert_correct_number_of_results(4, 0, input_data[0], result, 1)
    # todo assert output correct
    # assertions
    #assert result[0][-1] is None
