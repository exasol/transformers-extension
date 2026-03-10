"""
Protocol for using transformers prediction tasks in an udf.
Needs to be implemented for each task.
"""

from abc import (
    abstractmethod,
)
from collections.abc import Iterator
from typing import (
    Any,
    Protocol,
)

import pandas as pd


class PredictionTask(Protocol):
    """
    Any prediction tasks needs to implement these methods
        - create_dataframes_from_predictions
        - extract_unique_param_based_dataframes
        - execute_prediction
        - append_predictions_to_input_dataframe
    """

    def create_dataframes_from_predictions(
        self, predictions: list[Any]
    ) -> list[pd.DataFrame]:
        """
        create_dataframes_from_predictions` : Converts list of predictions to
        pandas dataframe.
        """
        pass

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """
        `extract_unique_param_based_dataframes` : Even if the data in a given
        dataframe all have the same model, there might be differences within the given
        dataframe with different model parameters.
        This method is responsible for extracting unique dataframes which share both the
        same model and model parameters.
        """
        pass

    def execute_prediction(self, model_df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        execute_prediction` : Performs prediction on a given text list using
        recently loaded models.
        """
        pass

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        append_predictions_to_input_dataframe`: Reformats the dataframe used in
        prediction, such that each input row has a row for each prediction result.
        """
        pass

