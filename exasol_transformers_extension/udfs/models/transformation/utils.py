from pandas import DataFrame


def _create_new_empty_columns(model_df: DataFrame, new_columns) -> DataFrame:
    """
    create new columns and fill with None
    """
    model_df[new_columns] = None
    return model_df


def _drop_old_columns(
    model_df: DataFrame, removed_columns: str | list[str]
) -> DataFrame:
    """
    drop old columns
    """
    res_df = model_df.drop(columns=removed_columns)
    return res_df


def _ensure_output_format(
    batch_df: DataFrame, new_columns, removed_columns
) -> DataFrame:
    """
    ensure all promised output columns are present
    """
    for new_column in new_columns:
        if new_column not in batch_df.columns:
            batch_df = _create_new_empty_columns(batch_df, new_column)
    for del_col in removed_columns:
        if del_col in batch_df.columns:
            batch_df = _drop_old_columns(batch_df, del_col)

    return batch_df


def _check_input_format(
    df_columns: list[str], expected_input_columns, transformation_name: str
):
    """
    checks if all needed columns for
    transform are present, throws error otherwise
    """
    if not all(col in df_columns for col in expected_input_columns):
        raise ValueError(
            f"Missing expected input columns for {transformation_name}. "
            f"Expected at least the following columns: {expected_input_columns} "
            f"got these input columns: {df_columns}"
        )
