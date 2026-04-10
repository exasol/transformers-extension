from collections.abc import Iterator

from pandas import DataFrame

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.utils import (
    _check_input_format,
    _ensure_output_format,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class UniqueModelParamsDataframeTransformation(Transformation):
    """
    Transformation which splits the input DataFrame into multiple DataFrames,
    based on which model-parameters are found.
    Calls PredictionTask.extract_unique_param_based_dataframes, since the
    model-parameters are tied to the transformers task-type.
    Fails if one of the columns is empty for a given row.
    """

    def __init__(
        self,
        prediction_task: PredictionTask,
        expected_input_columns: list[str],
        new_columns: list[str],
        removed_columns: list[str],
    ):
        self.prediction_task = prediction_task
        self.expected_input_columns = expected_input_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def transform(
        self, model_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        calls transformation logic.
        """
        yield from self.prediction_task.extract_unique_param_based_dataframes(model_df)

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
