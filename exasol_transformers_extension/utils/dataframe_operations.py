import pandas as pd
from typing import List, Any


def get_unique_values(
        df: pd.DataFrame,
        columns: List[str],
        sort: bool = False) -> List[List[Any]]:
    """
    Filter given dataframe and return the unique values. Sorts the unique values
    if the sort parameter is set to True

    :param df: Dataframe from which the unique values are extracted
    :param columns: List of columns to be filtered and sorted accordingly
    :param sort: Sort the unique values by given columns if True
    """

    unique_df = df[columns].drop_duplicates()
    if sort:
        unique_df.sort_values(by=columns, ascending=True, inplace=True)

    return unique_df.values.tolist()


def sort_cell_values(
        df: pd.DataFrame, column: str, sep: str = ",") -> pd.DataFrame:
    """
    Sort separated values in each cell

    :param df: Dataframe containing the data to be processed.
    :param column: Column containing the cell values to be listed.
    :param sep: Separator of values in cell
    """

    df[column] = df[column].apply(
        lambda cell: ','.join(sorted(cell.split(sep))))

    return df


