import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.fill_mask import (
    FillMaskPredictionTask,
)


class AiFillMaskExtendedUDF(BaseModelUDF):
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
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            new_columns=["filled_text", "score", "rank", "error_message"],
        )
