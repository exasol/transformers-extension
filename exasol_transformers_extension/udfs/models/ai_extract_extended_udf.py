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
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            new_columns=[
                "start_pos",
                "end_pos",
                "word",
                "entity",
                "score",
                "error_message",
            ],
            work_with_spans=work_with_spans,
        )
