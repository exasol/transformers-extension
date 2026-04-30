import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
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
from exasol_transformers_extension.udfs.models.transformation.span_columns import (
    SpanColumnsZeroShotTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)

"""
UDF labeling a given text.
If given an input span as input columns
text_data_doc_id, text_data_char_begin, text_data_char_end, the span should
represent the entire input text_data. This udf is not equipped to
select substrings of the input text for classification based on the input span.
"""
from exasol_transformers_extension.udfs.models.prediction_tasks.zero_shot import (
    ZeroShotPredictionTask,
)


class AiClassifyExtendedUDF(BaseModelUDF):
    """
    UDF labeling a given text, selecting the label from a defined list of candidate labels.

    If given an input span as input columns
    text_data_doc_id, text_data_char_begin, text_data_char_end, the span should
    represent the entire input text_data. This udf is not equipped to
    select substrings of the input text for classification based on the input span.

    Needs to have "text_data", "candidate_labels" in the input.
    Will output to "label", "score", "rank".
    Does not use default values.

    Uses models compatible with the "zero-shot-classification" transformers task, and uses
    AutoModelForSequenceClassification to load said model.
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSequenceClassification,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=ZeroShotPredictionTask(
            desired_fields_in_prediction=["labels", "scores"]
        ),
        work_with_spans: bool = False,
    ):
        transformations: list[Transformation] = [
            UniqueModelDataframeTransformation(),
            UniqueModelParamsDataframeTransformation(
                prediction_task=prediction_task,
                expected_input_columns=["candidate_labels"],
                new_columns=[],
                removed_columns=[],
            ),
            WithModelTransformation(
                exa,
                PredictionTaskTransformation(
                    prediction_task=prediction_task,
                    new_columns=["label", "score", "rank"],
                    expected_input_columns=["text_data", "candidate_labels"],
                    removed_columns=[
                        "labels",
                        "scores",
                    ],  # get created and renamed, might need to be removed in case of errors
                ),
            ),
            RemoveColumnsTransformation(
                removed_columns=["device_id"],
            ),
        ]
        if work_with_spans:
            transformations.append(
                SpanColumnsZeroShotTransformation(
                    expected_input_columns=["text_data", "candidate_labels"],
                    new_columns=[],  # no new span so no new columns. we just return the input span
                    removed_columns=["text_data", "candidate_labels"],
                )
            )

        transformations_p = TransformationPipeline(transformations)

        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations_p,
        )
