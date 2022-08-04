import pandas as pd
from typing import List,  TypeVar

T = TypeVar('T')


def get_sorted_unique_values(
        df: pd.DataFrame, columns: List[T]) -> List[List[T]]:
    """
    Filter given dataframe and return the sorted unique values

    :param df: Dataframe from which the uniques values are extracted
    :param columns: List of columns to be filtered and sorted accordingly
    """

    return df[columns].drop_duplicates().sort_values(by=columns).values.tolist()
