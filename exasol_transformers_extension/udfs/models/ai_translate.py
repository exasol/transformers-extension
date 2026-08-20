"""
Default UDF for translating text. Will prompt the model with
"translate <source_language> to <target_language>: <text-data>" #todo update
"""

import transformers

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)
from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.translation import (
    TranslatePredictionTask,
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
from exasol_transformers_extension.udfs.models.transformation.transformation_pipeline import (
    TransformationPipeline,
)
from exasol_transformers_extension.udfs.models.transformation.with_model_transformation import (
    WithModelTransformation,
)


class AiTranslateUDF(BaseModelUDF):
    """
    Default UDF for translating text. Will prompt the model with
    "translate <source_language> to <target_language>: <text-data>" #todo update
    #todo which langs does the model support?
    Needs to have  "text_data".
    Other input will be pulled from default values.
    Will output to "translation_text".
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSeq2SeqLM,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TranslatePredictionTask(desired_fields_in_prediction=[]),
    ):
        transformations = TransformationPipeline(
            [
                AddDefaultColumnsTransformation(
                    new_columns=[
                        "device_id",
                        "bucketfs_conn",
                        "sub_dir",
                        "model_name",
                        "max_new_tokens",
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
                    expected_input_columns=[
                        "max_new_tokens",
                        "source_language",
                        "target_language",
                    ],
                    new_columns=[],
                    removed_columns=[],
                ),
                WithModelTransformation(
                    exa,
                    PredictionTaskTransformation(
                        prediction_task=prediction_task,
                        new_columns=["translation_text"],
                        expected_input_columns=[
                            "source_language",
                            "target_language",
                            "text_data",
                            "max_new_tokens",
                        ],
                        removed_columns=[],
                    ),
                ),
                RemoveColumnsTransformation(
                    removed_columns=[
                        "device_id",
                        "bucketfs_conn",
                        "sub_dir",
                        "model_name",
                        "max_new_tokens",
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
