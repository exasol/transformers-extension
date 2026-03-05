
import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.translation import TranslatePredictionTask


class AiTranslateExtendedUDF(BaseModelUDF):
    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSeq2SeqLM,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TranslatePredictionTask(
                desired_fields_in_prediction=[],
                new_columns=["translation_text", "error_message"]
            ),
    ):
        super().__init__(
            exa, batch_size, pipeline, base_model, tokenizer, prediction_task=prediction_task
        )

