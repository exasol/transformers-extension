"""a dummy implementation for the base udf. used for testing base udf functionality."""

from collections.abc import Iterator
from typing import (
    Any,
    Dict,
    List,
    Union,
)

import pandas as pd
import transformers

from exasol_transformers_extension.udfs.models.base_model_udf import BaseModelUDF
from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)


class DummyPredictionTask(PredictionTask):
    def __init__(self, desired_fields_in_prediction: list[str]):
        super().__init__()
        self.last_created_pipeline = None
        self.task_type = "dummy-task"
        self._desired_fields_in_prediction = desired_fields_in_prediction

    def extract_unique_param_based_dataframes(
        self, model_df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        yield model_df

    def execute_prediction(
        self, model_df: pd.DataFrame
    ) -> list[Union[dict[str, Any], list[dict[str, Any]]]]:
        input_data = list(model_df["input_data"])
        results = self.last_created_pipeline(input_data)
        return results

    def append_predictions_to_input_dataframe(
        self,
        model_df: pd.DataFrame,
        pred_df_list: list[pd.DataFrame],
        work_with_spans: bool = False,
    ) -> pd.DataFrame:

        model_df = model_df.reset_index(drop=True)
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)

        if work_with_spans:
            model_df = self.create_new_span_columns(model_df)
            model_df = self.drop_old_data_for_span_execution(model_df)
        return model_df

    def create_dataframes_from_predictions(
        self, predictions: list[Union[dict[str, Any], list[dict[str, Any]]]]
    ) -> list[pd.DataFrame]:
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame(result)
            results_df_list.append(result_df)
        return results_df_list

    def create_new_span_columns(self, model_df: pd.DataFrame) -> pd.DataFrame:
        model_df[["test_span_column_add"]] = "add_this"
        return model_df

    def drop_old_data_for_span_execution(self, model_df: pd.DataFrame) -> pd.DataFrame:
        model_df = model_df.drop(columns=["test_span_column_drop"])
        return model_df


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
        super().__init__(
            exa,
            batch_size,
            pipeline,
            base_model,
            tokenizer,
            prediction_task=prediction_task,
            new_columns=["answer", "score", "error_message"],
            work_with_spans=work_with_spans,
        )
