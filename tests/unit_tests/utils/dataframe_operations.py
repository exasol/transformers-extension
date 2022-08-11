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
    sorted_unique_values = dataframe_operations.get_sorted_unique_values(
        sample_df, columns)

    assert expected == sorted_unique_values