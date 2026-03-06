
import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.zero_shot import ZeroShotPredictionTask

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
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            new_columns=["label", "score", "rank", "error_message"],
            work_with_spans=work_with_spans,
        )

