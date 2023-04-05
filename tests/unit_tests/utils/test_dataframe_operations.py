import pytest
import pandas as pd
from typing import List
from exasol_transformers_extension.utils import dataframe_operations


sample_arr = [0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
sample_df = pd.DataFrame({
    'A': sample_arr,
    'B': sorted(sample_arr, reverse=True),
    'C': list(range(len(sample_arr))),
    'D': sample_arr,
    'E': [1] * len(sample_arr)
})


@pytest.mark.parametrize("description, columns, expected", [
    ("sorted_column_with_duplicates", ['A'],
        [[0], [1], [2], [3], [4], [5]]),
    ("reverse_sorted_column_with_duplicates", ['B'],
        [[0], [1], [2], [3], [4], [5]]),
    ("two_column_same_sorting", ['A', 'D'],
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
    ("range", ['C'],
        [[i] for i in range(len(sample_arr))]),
    ("two_columns_different_sorting", ['A', 'B'],
        [[0, 5],
        [1, 5],
        [2, 5],
        [3, 4], [3, 5],
        [4, 3], [4, 4],
        [5, 0], [5, 1], [5, 2], [5, 3]]),
])
def test_get_sorted_unique_values(
        description: str, columns: List[str], expected: List[List[int]]):
    sorted_unique_values = dataframe_operations.get_unique_values(
        sample_df, columns, sort=True)

    assert expected == sorted_unique_values


@pytest.mark.parametrize("description, columns, expected", [
    ("sorted_column_with_duplicates", ['A'],
        [[0], [1], [2], [3], [4], [5]]),
    ("reverse_sorted_column_with_duplicates", ['B'],
        [[5], [4], [3], [2], [1], [0]]),
    ("two_column_same_sorting", ['A', 'D'],
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
    ("range", ['C'],
        [[i] for i in range(len(sample_arr))]),
    ("two_columns_different_sorting", ['A', 'B'],
        [[0, 5],
        [1, 5],
        [2, 5],
        [3, 5], [3, 4],
        [4, 4], [4, 3],
        [5, 3], [5, 2], [5, 1], [5, 0]]),
])
def test_get_unique_values(
        description: str, columns: List[str], expected: List[List[int]]):
    unique_values = dataframe_operations.get_unique_values(
        sample_df, columns)

    assert expected == unique_values


@pytest.mark.parametrize("description, seperator, dataframe, expected", [
    ("sort_empty_dataframe", ",", pd.DataFrame({"col": []}), []),
    ("sort_single_value", ",", pd.DataFrame({"col": ["A"]}), ["A"]),
    ("sort_column_values", ",", pd.DataFrame({"col": ["C,B,A"]}), ['A,B,C']),
    ("different_separator", ";", pd.DataFrame({"col": ["C;B;A"]}), ['A,B,C'])
])
def test_sorting_cell_values(
        description: str, seperator: str,
        dataframe: pd.DataFrame, expected: List[str]):

    dataframe = dataframe_operations.sort_cell_values(
        dataframe, 'col', seperator)
    assert list(dataframe['col'].values) == expected
