import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.text_classification import (
    TextClassifyPredictionTask,
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


class AiCustomClassifyUDF(BaseModelUDF):
    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModelForSequenceClassification,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=TextClassifyPredictionTask(desired_fields_in_prediction=[]),
    ):
        transformations = [
            UniqueModelDataframeTransformation(),
            UniqueModelParamsDataframeTransformation(
                prediction_task=prediction_task,
                expected_input_columns=[],
                new_columns=[],
                removed_columns=[],
            ),
            PredictionTaskTransformation(
                prediction_task=prediction_task,
                new_columns=["label", "score", "rank"],  # "error_message"]
            ),
        ]
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations,
        )
