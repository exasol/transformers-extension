import transformers

from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_MODEL_SPECS
from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
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
from exasol_transformers_extension.udfs.models.transformation.span_columns import (
    SpanColumnsZeroShotTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)

"""
Default UDF labeling a given text.
"""
from exasol_transformers_extension.udfs.models.prediction_tasks.zero_shot import (
    ZeroShotPredictionTask,
)


class AiClassifyUDF(BaseModelUDF):
    """
    UDF labeling a given text, selecting the label from a defined list of candidate labels.

    Needs to have "text_data", "candidate_labels" in the input.

    Other input will be pulled from default values.

    Will output to "label", "score", "rank".
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
    ):
        transformations: list[Transformation] = [
            AddDefaultColumnsTransformation(
                new_columns=[
                    "device_id",
                    "bucketfs_conn",
                    "sub_dir",
                    "model_name",
                    "return_ranks"
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
                expected_input_columns=["candidate_labels"],
                new_columns=[],
                removed_columns=[],
            ),
            WithModelTransformation(
                exa,
                PredictionTaskTransformation(
                    prediction_task=prediction_task,
                    new_columns=["label", "score", "rank"],
                    expected_input_columns=["text_data", "candidate_labels"],
                    removed_columns=[
                        "labels",
                        "scores",
                    ],  # get created and renamed, might need to be removed in case of errors
                ),
            ),
            RemoveColumnsTransformation(
                removed_columns=[
                    "device_id",
                    "bucketfs_conn",
                    "sub_dir",
                    "model_name",
                    "return_ranks"
                ],
            ),
        ]

        transformations_p = TransformationPipeline(transformations)

        super().__init__(
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations_p,
        )
