"""
Protocol for using transformers prediction tasks in an udf.
Needs to be implemented for each task.
"""

from collections.abc import Iterator
from typing import (
    Any,
    Protocol,
)

import pandas as pd


class PredictionTask(Protocol):
    """
    A PredictionTask implementation should be focused on using udf
    input and a transformers model to make a prediction.
    A PredictionTask should focus on using a specified transformers task-type
    and specific input/output columns.
    The input of a PredictionTask will always already be sorted into DataFrames which
    use the same model ( if model-parameters can be different, the sorting should
    be implemented in extract_unique_param_based_dataframes). The PredictionTask can also
    expect to receive all its input columns.
    If they are not already filled in the udf-input, they should be filled in a
    previous Transformation.

    Any prediction tasks needs to implement these methods
        - create_dataframes_from_predictions
        - extract_unique_param_based_dataframes
        - execute_prediction
        - append_predictions_to_input_dataframe
    """

    def create_dataframes_from_predictions(
        self, predictions: list[Any]
    ) -> Iterator[pd.DataFrame]:
        """
        Converts list of predictions to pandas dataframe.
        """
        pass

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """
        Even if the data in a given
        dataframe all have the same model, there might be differences within the given
        dataframe with different model parameters.
        This method is responsible for extracting unique dataframes which share both the
        same model and model parameters.
        """
        pass

    def execute_prediction(self, model_df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Performs prediction on a given text list using
        recently loaded models.
        """
        pass

    def append_predictions_to_input_dataframe(
        self, model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Reformats the model_df(used as input for the prediction),
        such that the generated predictions in pred_df_list can be merged with
        the model_df. Then performs the merge.
        This usually means repeating input rows as often as predictions where
        generated for that input-row.
        """
        pass
