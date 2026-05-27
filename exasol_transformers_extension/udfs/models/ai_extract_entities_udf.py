"""
Default UDF for finding and classifying a token/entity in a given text.
If given an input span, text_data_char_begin and text_data_char_end should
represent the entire input text and not indicate a substring.
"""

import transformers

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)
from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.token_classification import (
    TokenClassifyPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.add_default_columns import (
    AddDefaultColumnsTransformation,
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
from exasol_transformers_extension.udfs.models.transformation.remove_columns import (
    RemoveColumnsTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.span_columns import (
    SpanColumnsTokenClassificationTransformation,
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


class AiExtractEntitiesUDF(BaseModelUDF):
    """
    UDF for finding and classifying a token/entity in a given text.

    Needs to have "text_data" in the input.

    Other input will be pulled from default values.

    Will output to "word", "entity" and "score".
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
    ):
        transformations: list[Transformation] = [
            AddDefaultColumnsTransformation(
                new_columns=[
                    "device_id",
                    "bucketfs_conn",
                    "sub_dir",
                    "model_name",
                    "aggregation_strategy",
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
                expected_input_columns=["aggregation_strategy"],
                new_columns=[],
                removed_columns=[],
            ),
            WithModelTransformation(
                exa,
                PredictionTaskTransformation(
                    prediction_task=prediction_task,
                    new_columns=[
                        "start_pos",
                        "end_pos",
                        "word",
                        "entity",
                        "score",
                    ],
                    removed_columns=[
                        "start",
                        "end",
                        "entity_group",
                    ],  # this one might get created. it should then be renamed, but in case that fails we need to remove it
                    expected_input_columns=[
                        "text_data",
                        "aggregation_strategy",
                    ],
                ),
            ),
            RemoveColumnsTransformation(
                removed_columns=[
                    "device_id",
                    "bucketfs_conn",
                    "sub_dir",
                    "model_name",
                    "aggregation_strategy",
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
