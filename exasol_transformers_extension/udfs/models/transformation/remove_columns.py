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


class RemoveColumnsTransformation(Transformation):
    """
    Transformation removes given columns from a dataframe
    """

    def __init__(
        self,
        expected_input_columns: list[str],
        removed_columns: list[str],
    ):
        """
        :param expected_input_columns: List of expected input columns. Need to be at least removed_columns.
        :param removed_columns: List of columns to be removed from batch_df.
        """
        self.expected_input_columns = expected_input_columns
        self.new_columns = []
        self.removed_columns = removed_columns

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        remove columns from df
        """
        yield _drop_old_columns(batch_df, self.removed_columns)

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
        return _ensure_output_format(batch_df, self.new_columns, self.removed_columns)
