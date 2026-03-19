"""
Manages Calling DataFrame Transformations in the BaseModelUDf.
Transformation may be implemented to add new functionality.
"""

import traceback
from collections.abc import Iterator
from typing import Protocol

import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class Transformation(Protocol):
    """
    Any change which should be made to the input/prediction DataFrame of the UDF
    should be wrapped in a Transformation.
    These transformations are then given to the UDF as input, and executed in order.
    A transformation May change the number of rows it returns in a given UDF.
    It May also return multiple DataFrames, for example if it needs to split the input.
    All returned DataFrames should have the same columns.

    A Transformations __init__ should have the following inputs if the use of the
    transformation/utils function is desired.
        * expected_input_columns: list[str],
        * new_columns: list[str],
        * removed_columns: list[str],

    If you are implementing a new Model-UDF, make sure you actually
    need a new transformation. Your needs might already be covered with
    the PredictionTaskTransformation.
    """

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> list[DataFrame]:
        """
        Transformation logic.
        May throw errors in case the transformation fails.
        Error handling is then done in the TransformationGenerator.
        """
        pass

    def check_input_format(self, df_columns: list[str]) -> None:
        """
        Checks if all needed columns for transform are present, throws error otherwise.
        There is a basic implementation in transformation/utils you may use.
        """
        pass

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        """
        Ensure all promised output columns are present. Called in case transform fails,
        or a previous transformation failed.
        We need to ensure the output still has the right format, otherwise the udf
        will not emit any data.
        There is a basic implementation in transformation/utils you may use.
        """
        pass


class TransformationGenerator:
    """
    A Generator which wraps around a Transformation. Responsible for calling the
    transformation and managing error handling.
    Ensures the output has the correct format even if an error occurs, and the
    error-message is appended to the data correctly
    """

    def __init__(self, transformation: Transformation, model_loader: LoadLocalModel):
        """
        :param transformation: a Transformation which should be called by the generator.
        :param model_loader: a model-loader of class LoadLocalModel,
        used by a with_model_transformation to load the required model.
        Might be ignored if the Transformation does not use a model.
        """
        self._transformation = transformation
        self._model_loader = model_loader

    def transform(self, dfs: Iterator[DataFrame]) -> Iterator[DataFrame]:
        """
        Calls the transformation.transform and manages error handling.
        Ensures the output has the correct format even if an error occurs, and the
        error-message is appended to the data correctly
        """
        for df in dfs:
            if "error_message" in df.columns:
                correct_format_df = self._transformation.ensure_output_format(df)
                yield correct_format_df
                continue
            try:
                self._transformation.check_input_format(df.columns)
                yield from self._transformation.transform(df, self._model_loader)
            except Exception:
                stack_trace = traceback.format_exc()
                try:
                    correct_format_df = self._transformation.ensure_output_format(df)
                    result_with_error_df = self.get_result_with_error(
                        correct_format_df, stack_trace
                    )
                    yield result_with_error_df
                except Exception:
                    stack_trace_2 = traceback.format_exc()
                    result_with_error_df = self.get_result_with_error(df, stack_trace_2)
                    yield result_with_error_df

    def get_result_with_error(
        self, model_df: pd.DataFrame, stack_trace: str
    ) -> pd.DataFrame:
        """
        Add the stack trace to the dataframe that received an error
        during transformation.

        :param model_df: The dataframe that received an error during transformation
        :param stack_trace: String of the stack traceback
        """
        self.error_message_last(model_df)
        model_df["error_message"] = stack_trace
        return model_df

    @staticmethod
    def error_message_last(df: pd.DataFrame) -> pd.DataFrame:
        """
        in case there is an "error_message" column in the given Dataframe, sorts the
        columns so "error_message" is the last column.
        """
        cols = df.columns.tolist()
        if "error_message" in cols:
            # move error message column to the end of the df
            cols.remove("error_message")
            cols.append("error_message")
            df = df[cols]
        return df
