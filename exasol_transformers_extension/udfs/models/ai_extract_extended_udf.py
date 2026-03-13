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
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_param_dfs import \
    UniqueModelParamsDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation
from exasol_transformers_extension.udfs.models.transformation.span_columns import \
    SpanColumnsTokenClassificationTransformation


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
        transformations = [UniqueModelDataframeTransformation(),
                           UniqueModelParamsDataframeTransformation(
                               prediction_task=prediction_task),
                           PredictionTaskTransformation(
                               prediction_task=prediction_task,
                               new_columns=[
                                   "start_pos",
                                   "end_pos",
                                   "word",
                                   "entity",
                                   "score",
                                   #"error_message",
                               ]
                           )]
        if work_with_spans:
            transformations.append(SpanColumnsTokenClassificationTransformation())
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations
        )
