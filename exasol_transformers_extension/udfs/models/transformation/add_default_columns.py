from collections.abc import Iterator

from pandas import DataFrame

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
    DEFAULT_VALUES,
)
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.utils import (
    _check_input_format,
    _ensure_output_format,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class AddDefaultColumnsTransformation(Transformation):
    """
    Transformation adds default columns filled with default values to input dataframe.
    Will overwrite existing columns with same name with default values.

    You can overwrite the default values in DEFAULT_VALUES by passing a dict with
    different default values. You can also add columns not present in
    DEFAULT_VALUES in this dict.

    If a column should be added, but no default_value is found for the column, this column
    and all columns after this column will be added, but will be filled with "None",
    and an error will be added to "error_message"
    """

    def __init__(
        self,
        new_columns: list[str],
        udf_name: str,
        default_values: dict[str, str] = None,
    ):
        """
        :param new_columns: Names of the columns to be added to batch_df. Will throw KeyError if column name not known.
        :param udf_name: Name of the calling UDF class "Ai<name>UDF".
                         Used to decide which default model to load if model_name is in new_columns.
        :param default_values: Default values for column/value pairs not present in DEFAULT_VALUES.
        """
        self.expected_input_columns = []
        self.new_columns = new_columns
        self.removed_columns = []
        self.udf_name = udf_name
        self.default_values = default_values

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        add default columns filled with default values to input dataframe.
        """
        for default_column in self.new_columns:

            if default_column == "model_name":
                default_model_specs = DEFAULT_MODEL_SPECS[self.udf_name]
                batch_df[default_column] = default_model_specs.model_name
            elif (
                self.default_values is not None
                and default_column in self.default_values.keys()
            ):
                batch_df[default_column] = self.default_values[default_column]

            else:
                batch_df[default_column] = DEFAULT_VALUES[default_column]
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
        return _ensure_output_format(batch_df, self.new_columns, self.removed_columns)
