"""
UDF for continuing a given text. May be used to return whole text,
or only the newly generated part.
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.text_generation import (
    TextGenPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation


class AiCompleteExtendedUDF(BaseModelUDF):
    """
    UDF for continuing a given text. May be used to return whole text,
    or only the newly generated part.
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForCausalLM,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TextGenPredictionTask(
            desired_fields_in_prediction=[],
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
            new_columns=["generated_text", "error_message"],
        )
