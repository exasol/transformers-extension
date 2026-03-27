"""
Task logic for using the "fill-mask" transformers task in a prediction udf.
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
    create_rank_from_score,
    duplicate_input_rows_for_n_outputs, extract_unique_param_based_dataframes_on_col_list,
)


class FillMaskPredictionTask(PredictionTask):
    """
    Task logic for using the "fill-mask" transformers task in a prediction udf.
    """

    def __init__(
        self,
        desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "fill-mask"
        self._desired_fields_in_prediction = desired_fields_in_prediction
        self._mask_token = "<mask>"

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        yield from extract_unique_param_based_dataframes_on_col_list(model_df, ["top_k"])

    def execute_prediction(self, model_df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and filled texts

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        top_k = int(model_df["top_k"].iloc[0])
        text_data_raw = list(model_df["text_data"])
        text_data_with_valid_mask_token = self._get_text_data_with_valid_mask_token(
            text_data_raw
        )
        results = self.last_created_pipeline(
            text_data_with_valid_mask_token, top_k=top_k
        )

        #  Batch prediction returns list of list while single prediction just
        #  return a list. In order to ease dataframe operations, convert single
        #  prediction to list of list.
        results = [results] if len(text_data_raw) == 1 else results
        return results

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """
        model_df, pred_df = duplicate_input_rows_for_n_outputs(model_df, pred_df_list)
        # Concat predictions and model_df
        model_df = pd.concat([model_df, pred_df], axis=1)
        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: prediction results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame(result)
            result_df = result_df[self._desired_fields_in_prediction].rename(
                columns={"sequence": "filled_text"}
            )
            result_df = create_rank_from_score(result_df)
            results_df_list.append(result_df)
        return results_df_list

    def _get_text_data_with_valid_mask_token(
        self, text_data_raw: list[str]
    ) -> list[str]:
        """
        Replace user provided mask tokens with valid ones
        """
        return [
            text_data.replace(
                self._mask_token, self.last_created_pipeline.tokenizer.mask_token
            )
            for text_data in text_data_raw
        ]
