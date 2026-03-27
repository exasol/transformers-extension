"""
Task logic for using the "token-classification" transformers task in a prediction udf.
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
    duplicate_input_rows_for_n_outputs, extract_unique_param_based_dataframes_on_col_list,
)
from exasol_transformers_extension.utils import dataframe_operations


class TokenClassifyPredictionTask(PredictionTask):
    """
    Task logic for using the "token-classification" transformers task in a prediction udf.
    """

    def __init__(
        self,
        desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "token-classification"
        self._desired_fields_in_prediction = desired_fields_in_prediction
        self._default_aggregation_strategy = "simple"

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same aggregation_strategy
        parameter values

        :param model_df: Dataframe used in prediction

         :return: Unique model dataframes having same specified parameters
        """
        model_df["aggregation_strategy"] = model_df["aggregation_strategy"].fillna(
            self._default_aggregation_strategy
        )

        yield from extract_unique_param_based_dataframes_on_col_list(model_df, ["aggregation_strategy"])

    def execute_prediction(self, model_df: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores, entities and associated words

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        text_data = list(model_df["text_data"])
        aggregation_strategy = model_df["aggregation_strategy"].iloc[0]
        results = self.last_created_pipeline(
            text_data, aggregation_strategy=aggregation_strategy
        )

        results = results if isinstance(results[0], list) else [results]

        if aggregation_strategy == "none":
            self._desired_fields_in_prediction = [
                "start",
                "end",
                "word",
                "entity",
                "score",
            ]
        else:
            self._desired_fields_in_prediction = [
                "start",
                "end",
                "word",
                "entity_group",
                "score",
            ]

        return results

    def append_predictions_to_input_dataframe(
        self, model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame]
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
        model_df = pd.concat(
            [model_df, pred_df], axis=1, join="inner"
        )  # join='inner' -> drop rows where results are empty

        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[pd.DataFrame]:
        """
        Convert predictions to dataframe. Only score and answer fields are
        presented.

        :param predictions: predictions results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            if result and result[0]:
                result_df = pd.DataFrame(result)
                # need to save before trying to rename,
                # otherwise they get lost and cant be printed in error message
                result_df_column_names = result_df.columns
                try:
                    result_df = result_df[self._desired_fields_in_prediction].rename(
                        columns={
                            "start": "start_pos",
                            "end": "end_pos",
                            "entity_group": "entity",
                        }
                    )
                except KeyError as e:
                    # adding more detailed error message
                    raise KeyError(
                        f"Some expected column was not found in prediction results. "
                        f"Expected columns are: {self._desired_fields_in_prediction}. "
                        f"Prediction results contain columns: {result_df_column_names}"
                    ) from e
            else:
                # if the result for an input is empty, just append an empty result df,
                # and the input will be dropped during concatenation
                # we need to keep an empty dataframe, to make sure we have the same
                # amount of resul_df's in our list as we have input rows.
                # this way merging the df's later works smoothly.
                result_df = pd.DataFrame({})
            results_df_list.append(result_df)

        return results_df_list
