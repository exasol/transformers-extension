from typing import List, Tuple, Any, Optional

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
    def __init__(self, output: Output, n_input_columns: int,
                 error_message_index: int = -1):
        self.output = output
        self._n_input_columns = n_input_columns
        self._error_message_index = error_message_index

    def error_exists(self, row) -> Optional[bool]:
        return bool(row[self._error_message_index])

    def error_message(self, row) -> Optional[str]:
        return row[self._error_message_index]

    def input_columns(self, row) -> Tuple[Any]:
        return row[: self._n_input_columns]

    def __repr__(self):
        return repr(self.output)

    def __eq__(self, other):
        return all(
            row == output
            if not self.error_exists(row)
            else self.error_message(output) in self.error_message(row)
            and self.input_columns(row) == self.input_columns(output)
            for row, output in zip(self.output.data, other.data)
        )
