import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation
from exasol_transformers_extension.udfs.models.transformation.span_columns import SpanColumnsZeroShotTransformation

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
    UDF labeling a given text.
    If given an input span as input columns
    text_data_doc_id, text_data_char_begin, text_data_char_end, the span should
    represent the entire input text_data. This udf is not equipped to
    select substrings of the input text for classification based on the input span.
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
        transformations = [UniqueModelDataframeTransformation(),
                           PredictionTaskTransformation(prediction_task=prediction_task)]
        if work_with_spans:
            transformations.append(SpanColumnsZeroShotTransformation())

        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
            new_columns=["label", "score", "rank", "error_message"],
        )
