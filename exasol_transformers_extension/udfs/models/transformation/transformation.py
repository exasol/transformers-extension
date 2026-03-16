from typing import Protocol

from pandas import DataFrame


class Transformation(Protocol):
    # needs input and output columns specified


    def transform(self, batch_df: DataFrame) -> list[DataFrame]:
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

    def needs_model(self) -> bool:
        """
        return a bool indicating if a model is needed for this transformation
        #todo better suggestions?
        """
        pass
