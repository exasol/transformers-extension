"""a dummy implementation for the base udf. used for testing base udf functionality."""

from typing import (
    Any,
    Union,
)

import pandas as pd
import transformers
from pandas import DataFrame

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_param_dfs import \
    UniqueModelParamsDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation
from exasol_transformers_extension.udfs.models.transformation.utils import _ensure_output_format, _check_input_format, \
    _create_new_empty_columns, _drop_old_columns


class DummyPredictionTask(PredictionTask):
    def __init__(self, desired_fields_in_prediction: list[str]):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "dummy-task"
        self._desired_fields_in_prediction = desired_fields_in_prediction

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> list[pd.DataFrame]:
        return [model_df]

    def execute_prediction(
        self, model_df: pd.DataFrame
    ) -> list[Union[dict[str, Any], list[dict[str, Any]]]]:
        input_data = list(model_df["input_data"])
        results = self.last_created_pipeline(input_data)
        return results

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame]
    ) -> pd.DataFrame:

        model_df = model_df.reset_index(drop=True)
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)

        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[Union[dict[str, Any], list[dict[str, Any]]]]
    ) -> list[pd.DataFrame]:
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame(result)
            results_df_list.append(result_df)
        return results_df_list


class SpanColumnsDummyTransformation(Transformation):
    def __init__(
            self,
            expected_input_columns: list[str] = [],
            new_columns: list[str] = [],
            removed_columns: list[str] = []):
        # no new span so no new columns. we just return the input span
        new_columns = ["test_span_column_add"]#todo as input
        removed_columns = ["test_span_column_drop"]
        expected_input_columns = removed_columns
        self.expected_input_columns = expected_input_columns
        self.new_columns = new_columns
        self.removed_columns = removed_columns

    def transform(self, batch_df:DataFrame) -> list[DataFrame]:
        batch_df = _create_new_empty_columns(batch_df, self.new_columns)
        batch_df[self.new_columns] = "add_this"
        batch_df = _drop_old_columns(batch_df, self.removed_columns)
        return [batch_df]

    def check_input_format(self, df_columns:list[str]):
        """
        checks if all needed columns for
        transform are present, throws error otherwise
        """
        try:
            _check_input_format(df_columns,
                                self.expected_input_columns,
                                self.__class__.__name__)
        except Exception as e:
            raise e

    def ensure_output_format(self, batch_df:DataFrame) -> DataFrame:
        """
        ensure all promised output columns are present
        """
        return _ensure_output_format(batch_df, self.new_columns, self.removed_columns)




class DummyImplementationUDF(BaseModelUDF):
    """A dummy implementation for the  BaseModelUDF. used for testing BaseModelUDF functionality.
    implements necessary functions as simply as possible"""

    def __init__(
        self,
        exa,
        batch_size=100,
        pipeline=transformers.pipeline,
        base_model=transformers.AutoModel,
        tokenizer=transformers.AutoTokenizer,
        prediction_task=DummyPredictionTask(
            desired_fields_in_prediction=["answer", "score"]
        ),
        work_with_spans: bool = False,
    ):
        transformations = [UniqueModelDataframeTransformation(),
                           UniqueModelParamsDataframeTransformation(
                               prediction_task=prediction_task),
                           PredictionTaskTransformation(
                               prediction_task=prediction_task,
                               new_columns=["answer", "score",]
                           )]
        if work_with_spans:
            transformations.append(SpanColumnsDummyTransformation())
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            transformations=transformations
        )
