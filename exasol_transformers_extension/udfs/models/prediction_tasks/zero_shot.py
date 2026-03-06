from collections.abc import Iterator
from typing import (
    Any,
)

import pandas as pd
from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import PredictionTask
from exasol_transformers_extension.udfs.models.prediction_tasks.utils import select_result_on_return_rank
from exasol_transformers_extension.utils import dataframe_operations

class ZeroShotPredictionTask(PredictionTask):
    def __init__(
            self,
            desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type ="zero-shot-classification"
        self._desired_fields_in_prediction = desired_fields_in_prediction


    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same model parameter values. if there
        is no model specified parameter, the input dataframe return as it is.

        :param model_df: Dataframe used in prediction

        :return: Unique model dataframes having specified parameters
        """

        model_df_with_sorted_labels = dataframe_operations.sort_cell_values(
            model_df, "candidate_labels"
        )
        unique_params = dataframe_operations.get_unique_values(
            model_df_with_sorted_labels, ["candidate_labels"]
        )
        for candidate_label in unique_params:
            current_label = candidate_label[0]
            param_based_model_df = model_df[
                model_df["candidate_labels"] == current_label
            ]

            yield param_based_model_df

    def execute_prediction(self, model_df: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        sequences = list(model_df["text_data"])
        candidate_labels = model_df["candidate_labels"].iloc[0].split(",")
        results = self.last_created_pipeline(sequences, candidate_labels)
        return results

    def create_new_span_columns(self, model_df: pd.DataFrame) -> pd.DataFrame:
        # no new span so no new columns. we just return the input span
        return model_df

    def drop_old_data_for_span_execution(self, model_df: pd.DataFrame) -> pd.DataFrame:
        # drop columns which are made superfluous by the spans to save data transfer
        model_df = model_df.drop(columns=["text_data", "candidate_labels"])
        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[pd.DataFrame]:
        """
        Convert predictions to dataframe. If the prediction results can be
        presented as is, the results are converted directly into the dataframe.
        Otherwise, model-specific adjustments must be made in each model's
        own class.

        :param predictions: Prediction results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame(result)
            result_df = result_df[self._desired_fields_in_prediction].rename(
                columns={"labels": "label", "scores": "score"}
            )
            result_df["rank"] = (
                result_df["score"].rank(ascending=False, method="dense").astype(int)
            )
            results_df_list.append(result_df)

        return results_df_list

    def append_predictions_to_input_dataframe(
        self, model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame],
            work_with_spans: bool = False
    ) -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """
        merged_df_list = []
        for ix, pred_df in enumerate(pred_df_list):
            merged_df = pd.merge(model_df.iloc[[ix], :], pred_df, how="cross")
            merged_df = select_result_on_return_rank(merged_df)
            merged_df.reset_index()
            merged_df_list.append(merged_df)
        output_df = pd.concat(merged_df_list)

        if work_with_spans:
            output_df = self.create_new_span_columns(output_df)
            output_df = self.drop_old_data_for_span_execution(output_df)

        return output_df