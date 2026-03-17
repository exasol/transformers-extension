"""
UDF for finding and classifying a token/entity in a given text.
If given an input span, text_data_char_begin and text_data_char_end should
represent the entire input text and not indicate a substring.
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.token_classification import (
    TokenClassifyPredictionTask,
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
from exasol_transformers_extension.udfs.models.transformation.span_columns import (
    SpanColumnsTokenClassificationTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)


class AiExtractExtendedUDF(BaseModelUDF):
    """
    UDF for finding and classifying a token/entity in a given text.
    If given an input span, text_data_char_begin and text_data_char_end should
    represent the entire input text and not indicate a substring.
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForTokenClassification,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TokenClassifyPredictionTask(
            desired_fields_in_prediction=["start", "end", "word", "entity", "score"]
        ),
        work_with_spans: bool = False,
    ):
        transformations: list[Transformation] = [
            UniqueModelDataframeTransformation(),
            UniqueModelParamsDataframeTransformation(
                prediction_task=prediction_task,
                expected_input_columns=["aggregation_strategy"],
                new_columns=[],
                removed_columns=[],
            ),
            WithModelTransformation(
                exa,
                PredictionTaskTransformation(
                    prediction_task=prediction_task,
                    new_columns=[
                        "start_pos",
                        "end_pos",
                        "word",
                        "entity",
                        "score",
                    ],
                    removed_columns=[
                        "start",
                        "end",
                        "entity_group",
                    ],  # this one might get created. it should then be renamed, but in case that fais we need to remove it
                    expected_input_columns=[
                        "text_data",
                        "aggregation_strategy",
                    ],
                ),
            ),
        ]
        if work_with_spans:
            transformations.append(
                SpanColumnsTokenClassificationTransformation(
                    expected_input_columns=[
                        "text_data",
                        "start_pos",
                        "end_pos",
                        "word",
                        "entity",
                        "text_data_char_begin",
                        "text_data_doc_id",
                    ],
                    new_columns=[
                        "entity_doc_id",
                        "entity_char_begin",
                        "entity_char_end",
                    ],
                    removed_columns=["text_data", "start_pos", "end_pos"],
                )
            )
        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
        )
