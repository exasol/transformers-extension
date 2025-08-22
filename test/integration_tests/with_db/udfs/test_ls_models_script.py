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
    subdir = ""
    #subdir = str(model_params.sub_dir) # this is not set for most models in our params
    input_data = [[bucketfs_conn_name, subdir]]# todo do one with actuall subdir?
    expected_result = [] #todo
    input_data_subdir_not_exist = ["non-existend-subdir", bucketfs_conn_name, str(model_params.sub_dir)]
    input_data_subdir_empty = []

    query = (
        f"SELECT TE_LS_MODELS_UDF("
        f"t.bucketfs_conn_name, "
        f"t.sub_dir "
        f") FROM (VALUES {python_rows_to_sql(input_data)} "
        f"AS t(bucketfs_conn_name, "
        f"sub_dir));"
    )

    # execute UDF
    result = db_conn.execute(query).fetchall()
    # assertions
    assert result[0][-1] is None
    print(result)
    # added_columns: model_name, version, task_name, seed, path, error_message
    assert_correct_number_of_results(6, 0, input_data, result, 1)
    # todo assert output correct

'''
        test/integration_tests/with_db/udfs/test_ls_models_script.py::test_list_models_script[onprem] 

         [('TEST_TE_BFS_CONNECTION', 'container', 
         'exasol_transformers_extension_container-release-BGFZPNRUCX4GIGFCNHFHJ2RIB6BUBIUTLYNDSHPY2PMJWGTRDUCQ/usr/local/lib/python3.10/dist-packages/dateutil/zoneinfo/', 
         'dateutil-zoneinfo', 
         '/buckets/bfsdefault/default/container/exasol_transformers_extension_container-release-BGFZPNRUCX4GIGFCNHFHJ2RIB6BUBIUTLYNDSHPY2PMJWGTRDUCQ/usr/local/lib/python3.10/dist-packages/dateutil/zoneinfo/dateutil-zoneinfo.tar.gz', None), 
         
         ('TEST_TE_BFS_CONNECTION', 'container', 
         'exasol_transformers_extension_container-release-6SWLUZ6QB7PSXSU62OKB7AQXV6RNQPUIW3E3YH4QKXNZPET5MO5Q/usr/local/lib/python3.10/dist-packages/dateutil/zoneinfo/', 
         'dateutil-zoneinfo', 
         '/buckets/bfsdefault/default/container/exasol_transformers_extension_container-release-6SWLUZ6QB7PSXSU62OKB7AQXV6RNQPUIW3E3YH4QKXNZPET5MO5Q/usr/local/lib/python3.10/dist-packages/dateutil/zoneinfo/dateutil-zoneinfo.tar.gz', 
         None)]
'''