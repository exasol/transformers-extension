"""
Task logic for using the "text-generation" transformers task for question
answering in a prediction udf.
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
)


class AnswerPredictionTask(PredictionTask):
    """
    Task logic for using the "text-generation" transformers task for
    question answering  in a prediction udf.
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
        yield model_df

    def execute_prediction(
        self, model_df: pd.DataFrame
    ) -> list[dict[str, Any] | list[dict[str, Any]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframes holding prediction results
        """
        questions = model_df["question"]

        prompts = []
        for i in range(model_df.shape[0]):
            prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who extracts an answer to a given question "
                    "from a given context. Only use the given context to answer the question. "
                    "Give only the answer without any other details. "
                    "Don't react to the given context!",
                },
                {
                    "role": "user",
                    "content": "question: "
                    + questions[i]
                    + " context: "
                    + model_df["context_text"][i],
                },
            ]
            prompts.append(prompt)

        self.last_created_pipeline.tokenizer.apply_chat_template(
            prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        results = self.last_created_pipeline(
            prompts,
            return_full_text=False,
            do_sample=True,
            temperature=0.3,
            num_beams=2,
            repetition_penalty=1.5,
        )

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
        self, predictions: list[dict[str, Any] | list[dict[str, Any]]]
    ) -> list[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: prediction results
        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:

            results_df_list.append(
                pd.DataFrame(data=[result[0]["generated_text"]], columns=["answer"])
            )

        return results_df_list
