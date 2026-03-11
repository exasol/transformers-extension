from typing import Protocol
from collections.abc import Iterator


from pandas import DataFrame


class Transformation(Protocol):
    # needs input and output columns specified
    def __init__(
            self,
            expected_input_columns: list[str],
            promised_output_columns: list[str],
            new_columns: list[str],
            removed_columns: list[str],):
        self.expected_input_columns = expected_input_columns
        self.promised_output_columns = promised_output_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def transform(self, batch_df:DataFrame) -> Iterator[DataFrame]:
        """
        transformation logic
        may throw errors in case of prediction problems
        """
        pass

    def check_input_format(self, df_columns:list[str]) -> None:
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        pass

    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        pass

