import traceback
from abc import (
    ABC,
)
import numpy as np
import pandas as pd
import transformers

from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)

from exasol_transformers_extension.udfs.models.transformation.transformation import (
    Transformation,
)
from exasol_transformers_extension.utils import (
    device_management,
)

from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)


class BaseModelUDF(ABC):
    """
    This base class should be extended by each UDF class containing model logic.
    This class contains common operations for all prediction UDFs:
        - accesses data part-by-part based on predefined batch size
        - manages the model cache
        - reads the corresponding model from BucketFS into cache
        - creates model pipeline through transformer api
        - manages the creation of predictions and the preparation of results.


    If your UDF changes output format depending on work_with_spans,
    consider also implementing:
        - drop_old_data_for_span_execution
        - create_new_span_columns
    These can be used to help making sure df output format is correct even if an error
    occurs before the format is changed in the UDF itself

    """

    def __init__(
        self,
        batch_size: int,
        pipeline: transformers.Pipeline,
        base_model: ModelFactoryProtocol,
        tokenizer: ModelFactoryProtocol,
        prediction_task: PredictionTask,
        transformations: list[Transformation],
    ):
        self.batch_size = batch_size
        self.pipeline = pipeline
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = None
        self.model_loader = None
        self.prediction_task = prediction_task
        self.transformations = transformations

    def run(self, ctx):
        device_id = ctx.get_dataframe(1).iloc[0]["device_id"]
        self.device = device_management.get_torch_device(device_id)
        self.create_model_loader()
        ctx.reset()

        while True:
            batch_df = ctx.get_dataframe(num_rows=self.batch_size, start_col=1)
            if batch_df is None:
                break
            in_dfs = [batch_df]
            for transformation in self.transformations:
                transform_result_dfs = []
                for in_df in in_dfs:
                    if "error_message" in in_df:
                        correct_format_df = transformation.ensure_output_format(in_df)
                        transform_result_dfs.append(
                            correct_format_df
                        )  # todo make this not teccessary even after UniqueModelDataframeTransformation
                        in_dfs = transform_result_dfs
                        continue
                    try:
                        transformation.check_input_format(in_df.columns)
                        transform_result_dfs = (
                            transform_result_dfs + transformation.transform(in_df, self.model_loader)
                        )
                    except Exception:
                        stack_trace = traceback.format_exc()
                        try:
                            correct_format_df = transformation.ensure_output_format(
                                in_df
                            )
                            result_with_error_df = self.get_result_with_error(
                                correct_format_df, stack_trace
                            )
                            transform_result_dfs.append(result_with_error_df)
                        except Exception:
                            stack_trace_2 = traceback.format_exc()
                            result_with_error_df = self.get_result_with_error(
                                in_df, stack_trace_2
                            )
                            transform_result_dfs.append(result_with_error_df)

                in_dfs = transform_result_dfs
            result_dfs = []
            for df in in_dfs:
                if not "error_message" in df.columns:
                    df["error_message"] = None
                result_dfs.append(self.error_message_last(df))
            result_df = pd.concat(result_dfs)
            result_df = result_df.replace(np.nan, None)
            ctx.emit(result_df)

        self.model_loader.clear_device_memory()

    def create_model_loader(self):
        """
        Creates the model_loader.
        """
        self.model_loader = LoadLocalModel(
            pipeline_factory=self.pipeline,
            base_model_factory=self.base_model,
            tokenizer_factory=self.tokenizer,
            task_type=self.prediction_task.task_type,
            device=self.device,
        )

    @staticmethod
    def error_message_last(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns.tolist()
        if "error_message" in cols:
            # move error message column to the end of the df
            cols.remove("error_message")
            cols.append("error_message")
            df = df[cols]
        return df

    def get_result_with_error(
        self, model_df: pd.DataFrame, stack_trace: str
    ) -> pd.DataFrame:
        """
        Add the stack trace to the dataframe that received an error
        during prediction.

        :param model_df: The dataframe that received an error during prediction
        :param stack_trace: String of the stack traceback
        """
        BaseModelUDF.error_message_last(model_df)
        self.error_message_last(model_df)
        model_df["error_message"] = stack_trace

        return model_df
