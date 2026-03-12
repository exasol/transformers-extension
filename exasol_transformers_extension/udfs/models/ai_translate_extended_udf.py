"""
UDF for translating text. Will prompt the model with
"translate <source_language> to <target_language>: <text-data>"
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.translation import (
    TranslatePredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation


class AiTranslateExtendedUDF(BaseModelUDF):
    """
    UDF for translating text. Will prompt the model with
    "translate <source_language> to <target_language>: <text-data>"
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSeq2SeqLM,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TranslatePredictionTask(desired_fields_in_prediction=[]),
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
            new_columns=["translation_text", "error_message"],
        )
