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


# def extract_unique_param_based_dataframes(
#    model_df: pd.DataFrame
# ) -> Iterator[pd.DataFrame]:
#   """
#    Extract unique dataframes having same max_new_tokens, source_language,
#    and target_language parameter values

#    :param model_df: Dataframe used in prediction

#     :return: Unique model dataframes having same specified parameters
#    """
#    unique_column_names = ["max_new_tokens", "source_language", "target_language"]
#    unique_params = dataframe_operations.get_unique_values(
#        model_df, unique_column_names
#   )
#    for unique_param_set in unique_params:
# todo how to build this? if work use
#       param_based_model_df = model_df[
#           (model_df[unique_column_names[i]] == unique_param_set[i]) for
#           i in range(len(unique_param_set))
#            #& (model_df["source_language"] == source_language)
#            #& (model_df["target_language"] == target_language)
#        ]
#       model_df=param_based_model_df.copy()


# param_based_model_df = model_df[
#    (model_df["max_new_tokens"] == max_new_tokens)
#    & (model_df["source_language"] == source_language)
#    & (model_df["target_language"] == target_language)
# ]

#     yield param_based_model_df
