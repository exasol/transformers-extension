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
from exasol_transformers_extension.udfs.models.transformation.prediction_task import (
    PredictionTaskTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)


class AiAnswerExtendedUDF(BaseModelUDF):
    """
    UDF for answering a given "question" about a given "context_text"
    Needs to have "question", "context_text", "top_k" in the input.
    Will output to "answer", "score", "rank".
    Does not use default values.
    Uses models compatible with the "question-answering" transformers task,
     and uses AutoModelForQuestionAnswering to load said model.
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

        transformations = TransformationPipeline(
            [
                UniqueModelDataframeTransformation(),
                UniqueModelParamsDataframeTransformation(
                    prediction_task=prediction_task,
                    expected_input_columns=["top_k"],
                    new_columns=[],
                    removed_columns=[],
                ),
                WithModelTransformation(
                    exa,
                    PredictionTaskTransformation(
                        prediction_task=prediction_task,
                        expected_input_columns=["question", "context_text", "top_k"],
                        new_columns=["answer", "score", "rank"],
                        removed_columns=[],
                    ),
                ),
            ]
        )

        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task,
            transformations,
        )
