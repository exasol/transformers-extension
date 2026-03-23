"""
Udf for the "fill_mask" transformers task.
Will fill any occurence of "<mask>" in the input text data with a generated substring.
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.fill_mask import (
    FillMaskPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import (
    UniqueModelDataframeTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_param_dfs import (
    UniqueModelParamsDataframeTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.predicition_task import (
    PredictionTaskTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)


class AiFillMaskExtendedUDF(BaseModelUDF):
    """
    Udf for the generating replacement substrings for any "<mask>"-substring found in the
    input text. .

    Needs to have "top_k", "text_data" in the input.
    Will output to "filled_text", "score", "rank".
    Does not use default values.

    Uses models compatible with the "fill_mask" transformers task, and uses
    AutoModelForMaskedLM to load said model.
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
        transformations = [
            UniqueModelDataframeTransformation(),
            UniqueModelParamsDataframeTransformation(
                prediction_task=prediction_task,
                expected_input_columns=["top_k"],
                new_columns=[],
                removed_columns=[],
            ),
            WithModelTransformation(
                exa,
                PredictionTaskTransformation(
                    prediction_task=prediction_task,
                    new_columns=[
                        "filled_text",
                        "score",
                        "rank",
                    ],
                    expected_input_columns=["top_k", "text_data"],
                    removed_columns=[
                        "sequence"
                    ],  # this will be created and the renamed. if that fails we need to remove it
                ),
            ),
        ]
        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
        )
