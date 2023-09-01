from io import StringIO
from typing import List

import pandas as pd


class Result:
    def __init__(self, result_df: pd.DataFrame):
        self.result_df = result_df

    def __repr__(self):
        info_buffer = StringIO()
        self.result_df.info(buf=info_buffer)
        df_buffer = StringIO()
        self.result_df.to_string(buf=df_buffer)
        return f"Info:\n{info_buffer.getvalue()}\nDataFrame:\n{df_buffer.getvalue()}"


class ScoreMatcher:

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        is_score_typed_as_float = result_df['score'].dtypes == 'float'
        return is_score_typed_as_float

    def __repr__(self) -> str:
        return 'score: float'


class RankDTypeMatcher:

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        is_rank_typed_as_int = result_df['rank'].dtypes == 'int'
        return is_rank_typed_as_int

    def __repr__(self) -> str:
        return 'rank: int'


class RankMonotonicMatcher:

    def __init__(self, n_rows: int, results_per_row: int):
        self._n_rows = n_rows
        self._results_per_row = results_per_row

    def _is_rank_monotonic(self, score_rank_df: pd.DataFrame, row: int) -> bool:
        return (
            score_rank_df[row * self._results_per_row: self._results_per_row + row * self._results_per_row]
            .sort_values(by='score', ascending=False)['rank']
            .is_monotonic
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        is_rank_correct = \
            all([self._is_rank_monotonic(result_df, row)
                 for row in range(self._n_rows)])
        return is_rank_correct

    def __repr__(self) -> str:
        return 'rank is monotonic'


class ColumnsMatcher:

    def __init__(self, columns: List[str], new_columns: List[str]):
        self._new_columns = new_columns
        self._columns = columns
        self._expected_columns = self._columns + self._new_columns

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        has_valid_columns = result_df.columns == self._expected_columns
        return all(has_valid_columns)

    def __repr__(self) -> str:
        return str(self.__dict__)


class ShapeMatcher:

    def __init__(self, columns: List[str], new_columns: List[str], n_rows: int, results_per_row: int = 1):
        self._new_columns = new_columns
        self._columns = columns
        self._results_per_row = results_per_row
        self._n_rows = n_rows
        self._expected_shape = (self._n_rows * self._results_per_row, len(self._columns) + len(self._new_columns) - 1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        has_valid_shape = \
            result_df.shape == self._expected_shape
        return has_valid_shape

    def __repr__(self) -> str:
        return str(self.__dict__)


class NoErrorMessageMatcher:

    def __eq__(self, other):
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        is_error_message_none = not any(result_df['error_message'])
        return is_error_message_none

    def __repr__(self) -> str:
        return "no error_message"


class NewColumnsEmptyMatcher:

    def __init__(self, new_columns: List[str]):
        self._new_columns = new_columns

    def __eq__(self, other):
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        are_new_columns_none = all(
            all(result_df[col].isnull()) for col in self._new_columns[:-1]
        )
        return are_new_columns_none

    def __repr__(self):
        return f"{self._new_columns} are empty"


class ErrorMessageMatcher:

    def __eq__(self, other: Result):
        if not isinstance(other, Result):
            return False
        result_df = other.result_df
        has_valid_error_message = all(
            'Traceback' in row for row in result_df['error_message'])
        return has_valid_error_message

    def __repr__(self) -> str:
        return "Has valid error messages."
