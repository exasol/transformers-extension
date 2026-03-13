
from collections.abc import Iterator

from pandas import DataFrame

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import PredictionTask
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation


class UniqueModelParamsDataframeTransformation(Transformation):
    def __init__(
            self,
            prediction_task: PredictionTask,
            expected_input_columns: list[str] = [],
            promised_output_columns: list[str] = [],
            new_columns: list[str] = [],
            removed_columns: list[str] = [],
    ):
        self.prediction_task = prediction_task
        self.expected_input_columns = expected_input_columns #todo depend on pre task
        self.promised_output_columns = promised_output_columns #todo depend on pre task
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def needs_model(self) -> bool:
        return False

    def transform(self, model_df:DataFrame) -> Iterator[DataFrame]:
        result = self.prediction_task.extract_unique_param_based_dataframes(model_df)

        return result
         #yield self.extract_unique_model_dataframes_from_batch(batch_df)

    def check_input_format(self, df_columns:list[str]):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        if not all(col in df_columns for col in self.expected_input_columns):
            raise ValueError("Missing expected input columns for "
                             "UniqueModelDataframeTransformation. "
                             "Expected at least the following columns: %s"
                             "got these input columns: %s".format(
                self.expected_input_columns, df_columns))


    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        #todo
        return batch_df


