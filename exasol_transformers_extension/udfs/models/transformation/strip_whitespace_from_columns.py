from collections.abc import Iterator

from pandas import DataFrame

from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.utils import (
    _check_input_format,
    _drop_old_columns,
    _ensure_output_format,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class StripWhitespaceTransformation(Transformation):
    """
    Transformation strips whitespaces from entries of given columns
    """

    def __init__(
        self,
        expected_input_columns: list[str],
    ):
        """
        :param expected_input_columns: List of columns to be stripped
        """
        self.expected_input_columns = expected_input_columns

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        remove whitespaces from start/end of all strings in columns
        listed in expected_input_columns. replaces non-string entries with Nan
        """
        for col in self.expected_input_columns:
            batch_df[col] = batch_df[col].str.strip()
        yield batch_df

    def check_input_format(self, df_columns: list[str]):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        _check_input_format(
            df_columns, self.expected_input_columns, self.__class__.__name__
        )

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        return batch_df
