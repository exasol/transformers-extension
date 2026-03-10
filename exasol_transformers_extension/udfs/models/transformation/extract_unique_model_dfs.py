import traceback
from collections.abc import Iterator

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import PredictionTask
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification


class UniqueModelDataframeTransformation(Transformation):
    def __init__(
            self,
            expected_input_columns: list[str],
            promised_output_columns: list[str],
            new_columns: list[str],
            removed_columns: list[str],
    ):
        super().__init__(expected_input_columns,
                         promised_output_columns,
                         new_columns,
                         removed_columns)

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
    ) -> Iterator[pd.DataFrame]:
        """
        Extract unique model dataframes with the same model_name, bucketfs_conn,
        and sub_dir from the dataframe.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Unique model dataframe having same model_name,
        bucketfs_connection, and sub_dir
        """

        unique_values = dataframe_operations.get_unique_values(
            batch_df, constants.ordered_columns, sort=True
        )

        for model_name, bucketfs_conn, sub_dir in unique_values:
            try:
                self._check_values_not_null(model_name, bucketfs_conn, sub_dir)
            except ValueError:
                stack_trace = traceback.format_exc()
                result_with_error_df = self.get_result_with_error(batch_df, stack_trace)
                yield result_with_error_df
                return

            selections = (  # todo replace with specification in future?
                (batch_df["model_name"] == model_name)
                & (batch_df["bucketfs_conn"] == bucketfs_conn)
                & (batch_df["sub_dir"] == sub_dir)
            )

            model_df = batch_df[selections]

            yield model_df


    def transform(self, batch_df:DataFrame) -> DataFrame:
         return self.extract_unique_model_dataframes_from_batch(batch_df)

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


