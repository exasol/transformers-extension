"""
functions which get used in multiple PredictionTask implementations.
"""

from collections.abc import Iterator

import pandas as pd

from exasol_transformers_extension.utils import dataframe_operations


def extract_unique_param_based_dataframes_top_k(
    model_df: pd.DataFrame,
) -> Iterator[pd.DataFrame]:
    """
    Extract unique dataframes having same top_k parameter values

    :param model_df: Dataframe used in prediction

     :return: Unique model dataframes having specified parameters
    """
    unique_params = dataframe_operations.get_unique_values(model_df, ["top_k"])
    for top_k in unique_params:
        current_top_k = top_k[0]
        param_based_model_df = model_df[model_df["top_k"] == current_top_k]

        yield param_based_model_df


def duplicate_input_rows_for_n_outputs(
    model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeat each row consecutively as the number of entities. At the end,
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

