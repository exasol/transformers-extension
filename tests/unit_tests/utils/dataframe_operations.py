import pandas as pd
import pytest
from exasol_transformers_extension.utils import dataframe_operations


sample_arr = [0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
sample_df = pd.DataFrame({
    'A': sample_arr,
    'B': sorted(sample_arr, reverse=True),
    'C': list(range(len(sample_arr))),
    'D': sample_arr
})


@pytest.mark.parametrize("columns, expected", [
    (['A'], [[0], [1], [2], [3], [4], [5]]),
    (['B'], [[0], [1], [2], [3], [4], [5]]),
    (['A', 'D'], [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
    (['C'], [[i] for i in range(len(sample_arr))]),
    (['A', 'B'], [[0, 5],
                  [1, 5],
                  [2, 5],
                  [3, 4], [3, 5],
                  [4, 3], [4, 4],
                  [5, 0], [5, 1], [5, 2], [5, 3]]),
])
def test_get_sorted_unique_values(columns, expected):
    sorted_unique_values = dataframe_operations.get_sorted_unique_values(
        sample_df, columns)

    assert expected == sorted_unique_values
