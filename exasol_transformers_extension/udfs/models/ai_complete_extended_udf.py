"""
UDF for continuing a given text. May be used to return whole text,
or only the newly generated part.
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.text_generation import (
    TextGenPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import (
    UniqueModelDataframeTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_param_dfs import (
    UniqueModelParamsDataframeTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.prediction_task import (
    PredictionTaskTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.remove_columns import (
    RemoveColumnsTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)


class AiCompleteExtendedUDF(BaseModelUDF):
    """
    UDF for continuing a given text. Can be configured to do one or the other:
        * return the whole text ( input text + generated continuation )
        * return only the newly generated part

    Needs to have "text_data", "max_new_tokens", "return_full_text" in the input.
    Will output to "generated_text".
    Does not use default values.
    Uses models compatible with the "text-generation" transformers task, and uses
    AutoModelForCausalLM to load said model.
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
        transformations = TransformationPipeline(
            [
                UniqueModelDataframeTransformation(),
                UniqueModelParamsDataframeTransformation(
                    prediction_task=prediction_task,
                    expected_input_columns=["max_new_tokens", "return_full_text"],
                    new_columns=[],
                    removed_columns=[],
                ),
                WithModelTransformation(
                    exa,
                    PredictionTaskTransformation(
                        prediction_task=prediction_task,
                        new_columns=["generated_text"],
                        expected_input_columns=[
                            "text_data",
                            "max_new_tokens",
                            "return_full_text",
                        ],
                        removed_columns=[],
                    ),
                ),
                RemoveColumnsTransformation(
                    removed_columns=["device_id"],
                    expected_input_columns=["device_id"],
                ),
            ]
        )

        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
        )
