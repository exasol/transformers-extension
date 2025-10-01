from test.integration_tests.with_db.udfs.python_rows_to_sql import python_rows_to_sql
from test.unit.utils.utils_for_udf_tests import assert_result_matches_expected_output_order_agnostic
from test.utils.parameters import (
    model_params,
)


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


def test_list_models_script(setup_database, db_conn, upload_tiny_model_to_bucketfs):
    bucketfs_conn_name, _ = setup_database
    subdir = "model_sub_dir"
    model_specification = model_params.tiny_model_specs

    input_data = [(bucketfs_conn_name, subdir)]
    expected_result = [
        (
            bucketfs_conn_name,
            subdir,
            model_specification.model_name,
            model_specification.task_type,
            "/buckets/bfsdefault/default/container/"
            + str(upload_tiny_model_to_bucketfs),
            "WARNING: We found a model which was saved using a task_name we don't recognize.",
        )
    ]
    input_data_subdir_not_exist = [(bucketfs_conn_name, "non-existend-subdir")]
    expected_result_subdir_not_exist = [
        (bucketfs_conn_name, "non-existend-subdir", None, None, None, None)
    ]
    input_data_subdir_empty_string = [(bucketfs_conn_name, "")]
    expected_result_data_subdir_empty_string = [
        (bucketfs_conn_name, "None", None, None, None, None)
    ]

    test_data_sets = [
        (input_data, expected_result),
        (input_data_subdir_not_exist, expected_result_subdir_not_exist),
        (input_data_subdir_empty_string, expected_result_data_subdir_empty_string),
    ]

    for input_data_set, expected_result in test_data_sets:
        query = (
            f"SELECT TE_LIST_MODELS_UDF("
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
        # assertions
        assert_correct_number_of_results(4, 0, input_data[0], result, 1)
        assert_result_matches_expected_output_order_agnostic(
            result, expected_result, ["bucketfs_conn", "sub_dir"], sort_by_column=4
        )
