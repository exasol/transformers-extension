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

    @abstractmethod
    def create_dataframes_from_predictions(
        self, predictions: list[Any]
    ) -> list[pd.DataFrame]:
        pass

    @abstractmethod
    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        pass

    @abstractmethod
    def execute_prediction(self, model_df: pd.DataFrame) -> list[pd.DataFrame]:
        pass

    @abstractmethod
    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame],
        work_with_spans: bool = False,
    ) -> pd.DataFrame:
        pass

    def create_new_span_columns(self, model_df: pd.DataFrame) -> pd.DataFrame:
        return model_df

    def drop_old_data_for_span_execution(self, model_df: pd.DataFrame) -> pd.DataFrame:
        return model_df
