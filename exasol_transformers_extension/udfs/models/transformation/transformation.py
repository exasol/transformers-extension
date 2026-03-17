import traceback
from collections.abc import Iterator
from typing import Protocol

import pandas as pd
from pandas import DataFrame

from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class Transformation(Protocol):
    # needs input and output columns specified


    def transform(self, batch_df: DataFrame, model_loader: LoadLocalModel) -> list[DataFrame]:
        """
        transformation logic
        may throw errors in case of prediction problems
        """
        #todo model_loader into with model transform another way?
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

    def needs_model(self) -> bool:
        """
        return a bool indicating if a model is needed for this transformation
        #todo better suggestions?
        """
        pass
