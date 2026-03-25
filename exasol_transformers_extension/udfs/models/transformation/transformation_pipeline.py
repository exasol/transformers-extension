from collections.abc import Iterator

import numpy as np
from pandas import DataFrame

from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
    TransformationErrorHandler,
)
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel


class TransformationPipeline:
    """
    Pipeline holding transformations.
    """

    def __init__(self, transformations: list[Transformation]):
        self.transformations = transformations

    def execute(
        self, batch_df: DataFrame, model_loader: LoadLocalModel
    ) -> Iterator[DataFrame]:
        """
        Executes transformations.transform with error handling.
        """
        last_generator = iter([batch_df])
        for transformation in self.transformations:
            transformation_generator = TransformationErrorHandler(
                transformation, model_loader
            )
            current_generator = transformation_generator.transform(last_generator)
            last_generator = current_generator

        for df in last_generator:
            if not "error_message" in df.columns:
                df["error_message"] = None
            result_df = TransformationErrorHandler.error_message_last(df)
            yield result_df.replace(np.nan, None)
