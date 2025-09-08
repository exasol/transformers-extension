def assert_correct_number_of_results(
    added_columns: int,
    removed_columns: int,
    input_data_row: tuple,
    result: list,
    n_rows_result: int,
):
    n_cols_result = len(input_data_row) + (added_columns - removed_columns)
    assert len(result) == n_rows_result and len(result[0]) == n_cols_result, (
        f"format of result is not correct,"
        f"expected {n_rows_result} rows, {n_cols_result} columns."
        f"actual: {len(result)} rows, {len(result[0])} columns"
    )


def assert_correct_number_of_results_multiple_results_per_input(
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
