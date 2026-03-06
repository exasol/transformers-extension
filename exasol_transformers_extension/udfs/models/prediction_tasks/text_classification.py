"""
Task logic for using the "text-classification" transformers task in a prediction udf.
Two classes, one for one text input, one for two text inputs.
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
    duplicate_input_rows_for_n_outputs,
    select_result_on_return_rank,
)


def _extract_unique_param_based_dataframes(
    model_df: pd.DataFrame,
) -> Iterator[pd.DataFrame]:
    """
    Extract unique dataframes having same model parameter values. if there
    is no model specified parameter, the input dataframe return as it is.

    :param model_df: Dataframe used in prediction

    :return: Unique model dataframes having specified parameters
    """

    yield model_df


def _append_predictions_to_input_dataframe(
    model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame]
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
    model_df = select_result_on_return_rank(model_df)
    model_df.reset_index()
    return model_df


def _create_dataframes_from_predictions(
    predictions: list[list[dict[str, Any]]],
) -> list[pd.DataFrame]:
    """
    Convert predictions to dataframe. If the prediction results can be
    presented as is, the results are converted directly into the dataframe.
    Otherwise, model-specific adjustments must be made in each model's
    own class.

    :param predictions: Predictions results

    :return: List of prediction dataframes
    """
    results_df_list = []
    for result in predictions:
        result_df = pd.DataFrame(result)
        result_df["rank"] = (
            result_df["score"].rank(ascending=False, method="dense").astype(int)
        )
        results_df_list.append(result_df)

    return results_df_list


class EntailmentPredictionTask(PredictionTask):
    """
    Task logic for using the "text-classification" transformers task in a
    prediction udf.
    Expects two text inputs per row.
    """

    def __init__(
        self,
        desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "text-classification"
        self._desired_fields_in_prediction = desired_fields_in_prediction

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        return _extract_unique_param_based_dataframes(model_df)

    def execute_prediction(self, model_df: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        first_sequences = list(model_df["first_text"])
        second_sequences = list(model_df["second_text"])

        input_sequences = []
        for text, text_pair in zip(first_sequences, second_sequences):
            input_sequences.append({"text": text, "text_pair": text_pair})

        results = self.last_created_pipeline(input_sequences, top_k=None)

        return results

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame],
        work_with_spans: bool = False,
    ) -> pd.DataFrame:
        return _append_predictions_to_input_dataframe(model_df, pred_df_list)

    def create_dataframes_from_predictions(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[pd.DataFrame]:
        return _create_dataframes_from_predictions(predictions)


class TextClassifyPredictionTask(PredictionTask):
    """
    Task logic for using the "text-classification" transformers task in
    a prediction udf.
    Expects one text inputs per row.
    """

    def __init__(
        self,
        desired_fields_in_prediction: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "text-classification"
        self._desired_fields_in_prediction = desired_fields_in_prediction

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        return _extract_unique_param_based_dataframes(model_df)

    def execute_prediction(self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        sequences = list(model_df["text_data"])
        results = self.last_created_pipeline(sequences, top_k=None)
        return results

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame],
        work_with_spans: bool = False,
    ) -> pd.DataFrame:
        return _append_predictions_to_input_dataframe(model_df, pred_df_list)

    def create_dataframes_from_predictions(
        self, predictions: list[list[dict[str, Any]]]
    ) -> list[pd.DataFrame]:
        return _create_dataframes_from_predictions(predictions)
