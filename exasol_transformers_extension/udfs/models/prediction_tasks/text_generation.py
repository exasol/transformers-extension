"""
Task logic for using the "text-generation" transformers task in a prediction udf.
"""

from collections.abc import Iterator
from typing import (
    Any,
)

import pandas as pd

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)
from exasol_transformers_extension.udfs.models.prediction_tasks.utils import (
    extract_unique_param_based_dataframes_on_col_list,
)


class TextGenPredictionTask(PredictionTask):
    """
    Task logic for using the "text-generation" transformers task in a prediction udf.
    """

    def __init__(
        self,
        desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "text-generation"
        self._desired_fields_in_prediction = desired_fields_in_prediction

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same max_new_tokens and return_full_text
        parameter values

        :param model_df: Dataframe used in prediction

         :return: Unique model dataframes having same specified parameters
        """
        yield from extract_unique_param_based_dataframes_on_col_list(
            model_df, ["max_new_tokens", "return_full_text"]
        )

    def execute_prediction(self, model_df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: A tuple containing prediction score list and label list
        """
        text_data = list(model_df["text_data"])
        max_new_tokens = int(model_df["max_new_tokens"].iloc[0])
        return_full_text = bool(model_df["return_full_text"].iloc[0])
        results = self.last_created_pipeline(
            text_data, max_new_tokens=max_new_tokens, return_full_text=return_full_text
        )

        #  Batch prediction returns list of list while single prediction just
        #  return a list. In case of batch predictions, we need to flatten
        #  2D prediction results to 1D list
        results = sum(results, []) if isinstance((results[0]), list) else results
        return results

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a generated text

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df.reset_index(drop=True), pred_df], axis=1)

        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[dict[str, Any]]
    ) -> list[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: predictions results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            results_df_list.append(
                pd.DataFrame(
                    data=[result["generated_text"]], columns=["generated_text"]
                )
            )
        return results_df_list
