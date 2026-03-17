import traceback
from collections.abc import Iterator
from typing import Protocol

import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class Transformation(Protocol):
    # needs input and output columns specified

    def transform(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> list[DataFrame]:
        """
        transformation logic
        may throw errors in case of prediction problems
        """
        pass

    def check_input_format(self, df_columns: list[str]) -> None:
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        pass

    def ensure_output_format(self, batch_df: DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        pass


class TransformationGenerator:

    def __init__(self, transformation: Transformation, model_loader: LoadLocalModel):
        self._transformation = transformation
        self._model_loader = model_loader

    def transform(self, dfs: Iterator[DataFrame]) -> Iterator[DataFrame]:
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
        during prediction.

        :param model_df: The dataframe that received an error during prediction
        :param stack_trace: String of the stack traceback
        """
        self.error_message_last(model_df)
        model_df["error_message"] = stack_trace
        return model_df

    @staticmethod
    def error_message_last(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns.tolist()
        if "error_message" in cols:
            # move error message column to the end of the df
            cols.remove("error_message")
            cols.append("error_message")
            df = df[cols]
        return df
