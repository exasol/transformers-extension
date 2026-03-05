from collections.abc import Iterator
from typing import (
    Any,
)

import pandas as pd
import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_task import PredictionTask


class TextClassifyPredictionTask(PredictionTask):
    def __init__(
            self,
            desired_fields_in_prediction: list[str],
            new_columns: list[str],
    ):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "text-classification"
        self._desired_fields_in_prediction = desired_fields_in_prediction
        self.new_columns = new_columns

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same model parameter values. if there
        is no model specified parameter, the input dataframe return as it is.

        :param model_df: Dataframe used in prediction

        :return: Unique model dataframes having specified parameters
        """

        yield model_df

    def execute_prediction(
            self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
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
        self, model_df: pd.DataFrame, pred_df_list: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """
        n_labels = list(map(lambda x: x.shape[0], pred_df_list))
        repeated_indexes = model_df.index.repeat(repeats=n_labels)
        model_df = model_df.loc[repeated_indexes].reset_index(drop=True)

        # Concat predictions and model_df
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)
        # return all results for inputs with return_ranks == "ALL",
        # and only best(rank=1) result for inputs with return_ranks == "HIGHEST"
        model_df = model_df.query(
            '(return_ranks == "ALL") or ((rank == 1) and (return_ranks == "HIGHEST"))'
        )
        model_df.reset_index()
        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[list[dict[str, Any]]]
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

# todo update docu
class AiCustomClassifyUDF(BaseModelUDF):
    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSequenceClassification,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TextClassifyPredictionTask(
                desired_fields_in_prediction=[],
                new_columns= ["label", "score", "rank", "error_message"]
            ),
    ):
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task
        )
