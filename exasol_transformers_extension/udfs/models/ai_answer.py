"""
Default UDF for answering a given "question" about a given "context_text"
"""

import transformers

from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_MODEL_SPECS
from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.question_answering import (
    AnswerPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.add_default_columns import AddDefaultColumnsTransformation
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import (
    UniqueModelDataframeTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_param_dfs import (
    UniqueModelParamsDataframeTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.prediction_task import (
    PredictionTaskTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.remove_columns import (
    RemoveColumnsTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)


class AiAnswerUDF(BaseModelUDF):
    """
    Default UDF for answering a given "question" about a given "context_text"
    Needs to have "question", "context_text", in the input.
    Other input will be pulled from default values.
    Will output to "answer".
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForCausalLM,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=AnswerPredictionTask(
            desired_fields_in_prediction=["answer"],
        ),
    ):

        transformations = TransformationPipeline(
            [
                AddDefaultColumnsTransformation(
                    new_columns=[
                        "device_id",
                        "bucketfs_conn",
                        "sub_dir",
                        "model_name",
                    ],
                    default_values={
                        "model_name": DEFAULT_MODEL_SPECS[
                            self.__class__.__name__
                        ].model_name
                    },
                ),
                UniqueModelDataframeTransformation(),
                UniqueModelParamsDataframeTransformation(
                    prediction_task=prediction_task,
                    expected_input_columns=[],
                    new_columns=[],
                    removed_columns=[],
                ),
                WithModelTransformation(
                    exa,
                    PredictionTaskTransformation(
                        prediction_task=prediction_task,
                        expected_input_columns=["question", "context_text"],
                        new_columns=["answer"],
                        removed_columns=[],
                    ),
                ),
                RemoveColumnsTransformation(
                    removed_columns=[
                        "device_id",
                        "bucketfs_conn",
                        "sub_dir",
                        "model_name",
                    ],
                ),
            ]
        )

        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
        )
