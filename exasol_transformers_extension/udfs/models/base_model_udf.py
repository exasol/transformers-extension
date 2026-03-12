import traceback
from abc import (
    ABC,
)
from collections.abc import Iterator

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc
import numpy as np
import pandas as pd
import transformers

from exasol_transformers_extension.deployment.constants import constants
from exasol_transformers_extension.udfs.models.prediction_tasks.prediction_task import (
    PredictionTask,
)
from exasol_transformers_extension.udfs.models.transformation.extract_unique_model_dfs import \
    UniqueModelDataframeTransformation
from exasol_transformers_extension.udfs.models.transformation.predicition_task import PredictionTaskTransformation
from exasol_transformers_extension.udfs.models.transformation.span_columns import \
    SpanColumnsTokenClassificationTransformation
from exasol_transformers_extension.udfs.models.transformation.transformation import Transformation
from exasol_transformers_extension.utils import (
    dataframe_operations,
    device_management,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
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
        exa,
        batch_size: int,
        pipeline: transformers.Pipeline,
        base_model: ModelFactoryProtocol,
        tokenizer: ModelFactoryProtocol,
        prediction_task: PredictionTask,
        transformations: list[Transformation],
    ):
        self.exa = exa
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
            #transformations = [UniqueModelDataframeTransformation(),
            #                   PredictionTaskTransformation(prediction_task=self.prediction_task),
            #                   SpanColumnsTokenClassificationTransformation()]#as input
            in_dfs = [batch_df]
            for transformation in self.transformations:
                print("start :" + transformation.__class__.__name__)
                #todo append error messages instead of overwrite?
                transform_result_dfs = []
                for in_df in in_dfs:
                    with pd.option_context('display.max_rows', None, 'display.max_columns',
                                           None):  # more options can be specified also
                        print(in_df)
                    try:
                        #if "error_message" in in_df:
                        #    transform_result_dfs.append(in_df)  #todo make this not teccessary even after UniqueModelDataframeTransformation
                        #    continue
                        transformation.check_input_format(in_df.columns)
                        if transformation.needs_model():
                            # dont need to load new model if transform does not use a model
                            # todo in future pull model handling into seperate class?
                            self.check_cache(in_df) #todo this we now do a lot
                        transform_result_dfs = transform_result_dfs + transformation.transform(in_df)
                        #transform_result_dfs.append(transformation.transform(in_df))
                    except Exception:
                        stack_trace = traceback.format_exc()
                        print(stack_trace)
                        try:
                            in_df = transformation.ensure_output_format(in_df)
                        except Exception:
                            #todo what happens if we can not ensure output format? add to stacktrace?
                            # maybe append to error and hope udf can still emit ?
                            continue
                        result_with_error_df = self.get_result_with_error(in_df, stack_trace)
                        transform_result_dfs.append(result_with_error_df)

                    finally:
                        in_dfs = transform_result_dfs
                        print("finally:")
                        for df in in_dfs:
                            with pd.option_context('display.max_rows', None, 'display.max_columns',
                                                   None):  # more options can be specified also
                                print(df)

            result_df = pd.concat(in_dfs)
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

    def check_cache(self, model_df: pd.DataFrame) -> None:
        """
        If the model for the given dataframe is not cached, it is loaded into
        the cache before performing the prediction.

        :param model_df: Unique model dataframe having same model_name,
        bucketfs_connection, and sub_dir
        """
        model_name = model_df["model_name"].iloc[0]
        bucketfs_conn = model_df["bucketfs_conn"].iloc[0]
        sub_dir = model_df["sub_dir"].iloc[0]
        current_model_specification = BucketFSModelSpecification(
            model_name, self.prediction_task.task_type, bucketfs_conn, sub_dir
        )

        if self.model_loader.current_model_specification != current_model_specification:
            bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
                self.exa.get_connection(bucketfs_conn)
            )

            self.model_loader.clear_device_memory()
            self.model_loader.set_current_model_specification(
                current_model_specification
            )
            self.model_loader.set_bucketfs_model_cache_dir(bucketfs_location)

            try:
                self.prediction_task.last_created_pipeline = (
                    self.model_loader.load_models()
                )
            except Exception:
                stack_trace = traceback.format_exc()
                self.model_loader.last_model_loaded_successfully = False
                self.model_loader.model_load_error = stack_trace
                raise

        elif not self.model_loader.last_model_loaded_successfully:
            raise Exception(
                f"Model loading failed previously with : "
                f"{self.model_loader.model_load_error}"
            )


    def get_result_with_error(
        self, model_df: pd.DataFrame, stack_trace: str
    ) -> pd.DataFrame:
        """
        Add the stack trace to the dataframe that received an error
        during prediction.

        :param model_df: The dataframe that received an error during prediction
        :param stack_trace: String of the stack traceback
        """
        cols = model_df.columns.tolist()
        if "error_message" in cols:
            # move error message column to the end of the df
            cols.remove("error_message")
            cols.append("error_message")
            model_df = model_df[cols]
        model_df["error_message"] = stack_trace

        return model_df
