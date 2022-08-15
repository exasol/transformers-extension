import pandas as pd
from typing import List,  TypeVar

T = TypeVar('T')


def get_unique_values(
        df: pd.DataFrame,
        columns: List[T],
        sort: bool = False) -> List[List[T]]:
    """
    Filter given dataframe and return the unique values. Sorts the unique values
    if the sort parameter is set to True

    :param df: Dataframe from which the uniques values are extracted
    :param columns: List of columns to be filtered and sorted accordingly
    :param sort: Sort the unique values by given columns if True
    """

    unique_df = df[columns].drop_duplicates()
    if sort:
        unique_df.sort_values(by=columns, ascending=True, inplace=True)

    return unique_df.values.tolist()


