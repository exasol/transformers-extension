"""
UDF for answering a given "question" about a given "context_text"
"""

import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.question_answering import (
    AnswerPredictionTask,
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


class AiAnswerExtendedUDF(BaseModelUDF):
    """
    UDF for answering a given "question" about a given "context_text"
    """

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
        transformations = [
            UniqueModelDataframeTransformation(),
            UniqueModelParamsDataframeTransformation(prediction_task=prediction_task),
            PredictionTaskTransformation(
                prediction_task=prediction_task,
                new_columns=["answer", "score", "rank"],  # "error_message"]
            ),
        ]
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task,
            transformations,
        )
