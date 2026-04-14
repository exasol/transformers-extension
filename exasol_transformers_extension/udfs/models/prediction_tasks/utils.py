"""
functions which get used in multiple PredictionTask implementations.
"""

from collections.abc import Iterator

import pandas as pd

from exasol_transformers_extension.utils import dataframe_operations


def duplicate_input_rows_for_n_outputs(
    model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeat each row consecutively as the number of found predictions. At the end,
    the dataframe is expanded from (m, n) to (m*n_labels, n)
    """
    # n_labels can also represent n_entities or topk results
    n_labels = list(map(lambda x: x.shape[0], pred_df_list))
    repeated_indexes = model_df.index.repeat(repeats=n_labels)
    model_df = model_df.loc[repeated_indexes].reset_index(drop=True)

    pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
    return model_df, pred_df


def select_result_on_return_rank(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    return all results for inputs with return_ranks == "ALL",
    and only best(rank=1) result for inputs with return_ranks == "HIGHEST"
    """
    model_df = model_df.query(
        '(return_ranks == "ALL") or ((rank == 1) and (return_ranks == "HIGHEST"))'
    )
    return model_df


def create_rank_from_score(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the given dataframe by the "score" column and write
    the result to the "rank" column
    """
    result_df["rank"] = (
        result_df["score"].rank(ascending=False, method="dense").astype(int)
    )
    return result_df


def extract_unique_param_based_dataframes_on_col_list(
    model_df: pd.DataFrame, unique_column_names: list[str]
) -> Iterator[pd.DataFrame]:
    """
        Split model_df into subsets based on set of unique parameters found in the df.
        for the unique parameter sets, only columns in unique_column_names
        are taken into account.

        :param model_df: Dataframe used in prediction
        :param unique_column_names: list of column names which should be taken into account
                                    while splitting df

    #    :return: dataframes which contain rows from model_df, where all  columns
                  specified in unique_column_names contain the same value
    """
    unique_params = dataframe_operations.get_unique_values(
        model_df, unique_column_names
    )
    query_sting = ""
    for i in range(0, len(unique_column_names)):
        query_sting += (
            "`" + unique_column_names[i] + "` == @unique_param_set[" + str(i) + "] & "
        )

    for unique_param_set in unique_params:
        yield model_df.query(query_sting[:-2])
