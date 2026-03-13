import traceback
from collections.abc import Iterator
import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import PredictionTask
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation


class PredictionTaskTransformation(Transformation):
    def __init__(
            self,
            prediction_task: PredictionTask,
            expected_input_columns: list[str] = [],
            promised_output_columns: list[str] = [],
            new_columns: list[str] = [],
            removed_columns: list[str] = [],
):
        self.prediction_task = prediction_task
        self.expected_input_columns = expected_input_columns
        self.promised_output_columns = promised_output_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def needs_model(self) -> bool:
        return True

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
        return pred_df

    def get_prediction_from_unique_param_based_dataframes(
        self, param_based_model_df
    ) -> list[DataFrame]:
        """
        Performs separate predictions for data with the same parameters
        in the same model dataframe.

        :param model_df: Dataframe containing data that has the same model
        but can have different parameters.

        :return: List of prediction results
        """

        try:
            result_df = self.get_prediction(param_based_model_df)
            #yield result_df
        except Exception as err:
            raise err
        return [result_df]


    def transform(self, batch_df:DataFrame) -> Iterator[DataFrame]:
         return self.get_prediction_from_unique_param_based_dataframes(batch_df)
        #todo concat before return?

    def check_input_format(self, batch_df:DataFrame):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        #todo depends on task type?
        return batch_df

    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        #todo depends on task type?
        for col in self.new_columns:
            batch_df[col] = None
        return batch_df


