import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.utils import (
    _check_input_format,
    _ensure_output_format,
)
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class UniqueModelDataframeTransformation(Transformation):
    """
    Transformation which Splits the input DataFrame into multiple DataFrames,
    based on which model is found in the model_name, bucketfs_conn and sub_dir columns.
    Fails if one of the columns is empty for a given row.
    """

    def __init__(
        self,
    ):
        self.expected_input_columns = constants.ordered_columns
        self.new_columns = []
        self.removed_columns = []

    @staticmethod
    def _check_values_not_null(model_name, bucketfs_conn, sub_dir):
        """
        Fails if one of the model columns( model_name, bucketfs_conn and sub_dir)
        is empty for a given row.
        """
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
                raise err

            selections = (  # todo replace with specification in future?
                (batch_df["model_name"] == model_name)
                & (batch_df["bucketfs_conn"] == bucketfs_conn)
                & (batch_df["sub_dir"] == sub_dir)
            )

            model_df = batch_df[selections]
            result_dfs.append(model_df)
        return result_dfs

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> list[DataFrame]:
        """
        calls transformation logic.
        """
        return self.extract_unique_model_dataframes_from_batch(batch_df)

    def check_input_format(self, df_columns: list[str]):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        try:
            _check_input_format(
                df_columns, self.expected_input_columns, self.__class__.__name__
            )
        except Exception as e:
            raise e

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        return _ensure_output_format(batch_df, self.new_columns, self.removed_columns)
