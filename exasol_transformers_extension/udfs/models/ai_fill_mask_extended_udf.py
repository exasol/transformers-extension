"""
Udf for the "fill_mask" transformers task.
Will fill any occurence of "<mask>" in the input text data with a generated substring.
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.fill_mask import (
    FillMaskPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation


class AiFillMaskExtendedUDF(BaseModelUDF):
    """
    Udf for the "fill_mask" transformers task.
    Will fill any occurence of "<mask>" in the input text data with a generated substring.
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForMaskedLM,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=FillMaskPredictionTask(
            desired_fields_in_prediction=["sequence", "score"]
        ),
    ):
        transformations = [UniqueModelDataframeTransformation(),
                           PredictionTaskTransformation(prediction_task=prediction_task)]
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
            new_columns=["filled_text", "score", "rank", "error_message"],
        )
