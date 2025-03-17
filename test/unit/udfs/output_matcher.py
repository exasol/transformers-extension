from typing import List, Tuple, Any, Optional

Row = List[Tuple[Any, ...]]


class Output:
    """encapsulates a model output"""
    def __init__(self, data: Row):
        self._data = data

    def __repr__(self):
        return f"{self._data}"

    @property
    def data(self):
        return self._data


class OutputMatcher:
    """takes an (expected) Output and corresponding number of input columns (n_input_columns) and allows multiple
    query's to be made about the output. most importantly can check weather this Output equals another Output"""
    def __init__(self, output: Output, n_input_columns: int,
                 error_message_index: int = -1):
        self.actual_output = output
        self._n_input_columns = n_input_columns
        self._error_message_index = error_message_index

    def error_exists(self, row) -> Optional[bool]:
        """returns true if error exists in error column"""
        return row[self._error_message_index] is not None

    def error_message(self, row) -> Optional[str]:
        """returns error message from error column"""
        return row[self._error_message_index]

    def input_columns(self, row) -> Tuple[Any]:
        """returns only the part of the output from the first up until n_input_columns.
        the assumption being that those correspond to the input columns"""
        return row[: self._n_input_columns]

    def __repr__(self):
        return repr(self.actual_output)

    def __eq__(self, expected_output: Output) -> bool:
        result = all(
            self.compare_row(expected_row=expected, actual_row=actual_row)
            for actual_row, expected in
            zip(self.actual_output.data, expected_output.data)
        )
        return result

    def compare_row(self, expected_row, actual_row):
        """checks weather two rows are equal"""
        if not self.error_exists(expected_row):
            return expected_row == actual_row

        return (
                self.error_message(expected_row) in self.error_message(actual_row)
                and self.input_columns(expected_row) == self.input_columns(actual_row)
        )
