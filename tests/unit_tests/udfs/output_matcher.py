from typing import List, Tuple, Any, Dict, Optional

Row = List[Tuple[Any, ...]]


class Output:
    def __init__(self, data: Row):
        self._data = data

    def __repr__(self):
        return f"{self._data}"

    @property
    def data(self):
        return self._data


class OutputMatcher:
    def __init__(self, output: Output, index_map: Dict[str, int]):
        self.output = output
        self._ix_error_message = index_map["error_message_col_index"]
        self._ix_prediction = index_map["prediction_col_index"]
        self._ix_end_of_input_cols = index_map["end_of_input_col_index"]

    def error_exists(self, row) -> Optional[bool]:
        return row[self._ix_prediction]

    def error_message(self, row) -> Optional[str]:
        return row[self._ix_error_message]

    def input_columns(self, row) -> Tuple[Any]:
        return row[: self._ix_end_of_input_cols]

    def __repr__(self):
        return repr(self.output)

    def __eq__(self, other):
        return all(
            row == output
            if self.error_exists(row)
            else self.error_message(output) in self.error_message(row)
            and self.input_columns(row) == self.input_columns(output)
            for row, output in zip(self.output.data, other.data)
        )
