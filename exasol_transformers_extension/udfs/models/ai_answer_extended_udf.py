import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.question_answering import (
    AnswerPredictionTask,
)


class AiAnswerExtendedUDF(BaseModelUDF):
    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForQuestionAnswering,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=AnswerPredictionTask(
            desired_fields_in_prediction=["answer", "score"],
        ),
    ):
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task,
            new_columns=["answer", "score", "rank", "error_message"],
        )
