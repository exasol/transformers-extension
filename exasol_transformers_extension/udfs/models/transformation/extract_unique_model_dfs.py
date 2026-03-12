
from collections.abc import Iterator

import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation
from exasol_transformers_extension.utils import dataframe_operations

class UniqueModelDataframeTransformation(Transformation):
    def __init__(
            self,
            expected_input_columns: list[str] = [],
            promised_output_columns: list[str] = [],
            new_columns: list[str] = [],
            removed_columns: list[str] = [],
    ):
        expected_input_columns = constants.ordered_columns
        self.expected_input_columns = expected_input_columns
        self.promised_output_columns = promised_output_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def needs_model(self) -> bool:
        return False

    @staticmethod
    def _check_values_not_null(model_name, bucketfs_conn, sub_dir):
        if not (model_name and bucketfs_conn and sub_dir):
            error_message = (
                f"For each model model_name, bucketfs_conn and sub_dir need to be "
                f"provided. "
                f"Found model_name = {model_name}, bucketfs_conn = {bucketfs_conn}, sub_dir = {sub_dir}."
            )
            raise ValueError(error_message)

    def extract_unique_model_dataframes_from_batch(
        self, batch_df: pd.DataFrame
    ) -> list[pd.DataFrame]:
        """
        Extract unique model dataframes with the same model_name, bucketfs_conn,
        and sub_dir from the dataframe.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Unique model dataframe having same model_name,
        bucketfs_connection, and sub_dir
        """

        unique_values = dataframe_operations.get_unique_values(
            batch_df, self.expected_input_columns, sort=True
        )

        result_dfs = []
        for model_name, bucketfs_conn, sub_dir in unique_values:
            try:
                self._check_values_not_null(model_name, bucketfs_conn, sub_dir)
            except ValueError as err:
                raise err #todo how reraise?

            selections = (  # todo replace with specification in future?
                (batch_df["model_name"] == model_name)
                & (batch_df["bucketfs_conn"] == bucketfs_conn)
                & (batch_df["sub_dir"] == sub_dir)
            )

            model_df = batch_df[selections]
            result_dfs.append(model_df)
        return result_dfs
            #yield model_df


    def transform(self, batch_df:DataFrame) -> Iterator[DataFrame]:
        result = self.extract_unique_model_dataframes_from_batch(batch_df)
        print("res: ")
        print(result)
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
        #todo #we could fill expected input columns here if not present?
        pass


