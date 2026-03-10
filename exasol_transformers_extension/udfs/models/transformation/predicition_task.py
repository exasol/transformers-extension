import traceback

import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import PredictionTask
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation


class PredictionTaskTransformation(Transformation):
    def __init__(
            self,
            expected_input_columns: list[str],
            promised_output_columns: list[str],
            new_columns: list[str],
            removed_columns: list[str],
            prediction_task: PredictionTask):
        self.prediction_task = prediction_task
        super().__init__(expected_input_columns,
                         promised_output_columns,
                         new_columns,
                         removed_columns)


    def get_prediction(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform prediction of the given model and preparation of the prediction
        results according to the format that the UDF can emit.

        :param model_df: The dataframe to be predicted

        :return: The dataframe where the model_df is formatted with the
        prediction results
        """

        predictions = self.prediction_task.execute_prediction(model_df)
        pred_df_list = self.prediction_task.create_dataframes_from_predictions(
            predictions
        )
        pred_df = self.prediction_task.append_predictions_to_input_dataframe(
            model_df, pred_df_list
        )
        pred_df["error_message"] = None
        return pred_df

    def get_prediction_from_unique_param_based_dataframes(
        self, model_df
    ) -> list[pd.DataFrame]:
        """
        Performs separate predictions for data with the same parameters
        in the same model dataframe.

        :param model_df: Dataframe containing data that has the same model
        but can have different parameters.

        :return: List of prediction results
        """
        result_df_list = []
        for (
            param_based_model_df
        ) in self.prediction_task.extract_unique_param_based_dataframes(model_df):
            try:
                result_df = self.get_prediction(param_based_model_df)
                result_df_list.append(result_df)
            except Exception:
                stack_trace = traceback.format_exc()
                result_with_error_df = self.get_result_with_error(
                    param_based_model_df, stack_trace
                )
                result_df_list.append(result_with_error_df)
        return result_df_list


    def transform(self, batch_df:DataFrame) -> DataFrame:
         return self.get_prediction_from_unique_param_based_dataframes(batch_df)

    def check_input_format(self, batch_df:DataFrame):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        #todo
        pass

    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        #todo
        pass


