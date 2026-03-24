import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.text_classification import (
    EntailmentPredictionTask,
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


class AiEntailmentExtendedUDF(BaseModelUDF):
    """
    UDf for measuring the similarity of two input tet-sequences.

    Needs to have "first_text", "second_text" in the input.
    Will output to "label", "score", "rank".
    Does not use default values.

    Uses models compatible with the "text-classification" transformers task, and uses
    AutoModelForSequenceClassification to load said model.
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSequenceClassification,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=EntailmentPredictionTask(desired_fields_in_prediction=[]),
    ):
        transformations = [
            UniqueModelDataframeTransformation(),
            UniqueModelParamsDataframeTransformation(
                prediction_task=prediction_task,
                expected_input_columns=[],
                new_columns=[],
                removed_columns=[],
            ),
            WithModelTransformation(
                exa,
                PredictionTaskTransformation(
                    prediction_task=prediction_task,
                    new_columns=["label", "score", "rank"],
                    expected_input_columns=["first_text", "second_text"],
                    removed_columns=[],
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
