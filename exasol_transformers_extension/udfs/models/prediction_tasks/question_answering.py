from collections.abc import Iterator
from typing import (
    Any,
    Union,
)

import pandas as pd
from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import PredictionTask
from exasol_transformers_extension.udfs.models.prediction_tasks.utils import \
    extract_unique_param_based_dataframes_top_k, \
    duplicate_input_rows_for_n_outputs


class AnswerPredictionTask(PredictionTask):
    def __init__(
            self,
            desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "question-answering"
        self._desired_fields_in_prediction = desired_fields_in_prediction


    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        return extract_unique_param_based_dataframes_top_k(model_df)

    def execute_prediction(
        self, model_df: pd.DataFrame
    ) -> list[Union[dict[str, Any], list[dict[str, Any]]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframes holding prediction results
        """
        questions = list(model_df["question"])
        contexts = list(model_df["context_text"])
        top_k = int(model_df["top_k"].iloc[0])
        results = self.last_created_pipeline(
            question=questions, context=contexts, top_k=top_k
        )

        # We need to separate the answer to one question from the answers to
        # multiple questions, such that results of one question could be
        # - a dict where top_k=1, or
        # - either a dict or list of dicts where top_k>1
        # in both cases we need to put the answer(s) in a list to make sure that
        # the answer(s) is from a single question
        results = [results] if len(questions) == 1 else results
        return results

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
        model_df, pred_df = duplicate_input_rows_for_n_outputs(model_df, pred_df_list)
        # Concat predictions and model_df
        model_df = pd.concat([model_df, pred_df], axis=1)
        return model_df


    def create_dataframes_from_predictions(
        self, predictions: list[Union[dict[str, Any], list[dict[str, Any]]]]
    ) -> list[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: prediction results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            result_df = (
                pd.DataFrame([result]) if type(result) == dict else pd.DataFrame(result)
            )
            result_df = result_df[self._desired_fields_in_prediction]
            result_df["rank"] = (
                result_df["score"].rank(ascending=False, method="dense").astype(int)
            )
            results_df_list.append(result_df)

        return results_df_list