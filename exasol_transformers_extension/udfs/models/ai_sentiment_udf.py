"""
Default UDF class for classifying a given text sequence by sentiment.
"""

import transformers

from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)
from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.text_classification import (
    TextClassifyPredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.add_default_columns import (
    AddDefaultColumnsTransformation,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import (
    UniqueModelDataframeTransformation,
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


class AiSentimentUDF(BaseModelUDF):
    """
    UDF for classifying a given text sequence by sentiment.

    Needs to have "text_data" in the input.

    other input will be pulled from default values.

    Will output to "label", "score".
    """

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSequenceClassification,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TextClassifyPredictionTask(desired_fields_in_prediction=[]),
    ):
        transformations = TransformationPipeline(
            [
                AddDefaultColumnsTransformation(
                    new_columns=[
                        "device_id",
                        "bucketfs_conn",
                        "sub_dir",
                        "return_ranks",
                        "model_name",
                    ],
                    default_values={
                        "model_name": DEFAULT_MODEL_SPECS[
                            self.__class__.__name__
                        ].model_name
                    },
                ),
                UniqueModelDataframeTransformation(),
                WithModelTransformation(
                    exa,
                    PredictionTaskTransformation(
                        prediction_task=prediction_task,
                        new_columns=["label", "score", "rank"],
                        expected_input_columns=["text_data"],
                        removed_columns=[],
                    ),
                ),
                RemoveColumnsTransformation(
                    removed_columns=[
                        "device_id",
                        "bucketfs_conn",
                        "sub_dir",
                        "model_name",
                        "return_ranks",
                        "rank",
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
