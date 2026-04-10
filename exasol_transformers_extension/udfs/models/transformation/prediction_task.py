from collections.abc import Iterator

import pandas as pd
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


class PredictionTaskTransformation(Transformation):
    """
    Transformation which calls
        prediction_task.execute_prediction
        prediction_task.create_dataframes_from_predictions
        prediction_task.append_predictions_to_input_dataframe
    and returns a DataFrame containing input and  prediction results.
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

    def get_prediction(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform prediction of the given model and preparation of the prediction
        results according to the format that the UDF can emit.

        :param model_df: The dataframe to be predicted

        :return: The dataframe where the model_df is formatted with the
        prediction results
        """

        predictions = self.prediction_task.execute_prediction(model_df)
        pred_df_list = self.prediction_task.create_dataframes_from_predictions(
            predictions
        )
        pred_df = self.prediction_task.append_predictions_to_input_dataframe(
            model_df, pred_df_list
        )
        return pred_df

    def get_prediction_from_unique_param_based_dataframes(
        self, param_based_model_df
    ) -> Iterator[DataFrame]:
        """
        Performs separate predictions for data with the same parameters
        in the same model dataframe.

        :param param_based_model_df: Dataframe containing data that has the same model
        but can have different parameters.

        :return: List of prediction results
        """
        yield self.get_prediction(param_based_model_df)

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        calls transformation logic.
        """
        yield from self.get_prediction_from_unique_param_based_dataframes(batch_df)

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
