import os
from abc import abstractmethod, ABC
from typing import Iterator, List, Any
import torch
import traceback
import pandas as pd
import numpy as np
import transformers

from exasol_transformers_extension.deployment import constants
from exasol_transformers_extension.utils import device_management, \
    bucketfs_operations, dataframe_operations
from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.model_specification import ModelSpecification


class BaseModelUDF(ABC):
    """
    This base class should be extended by each UDF class containing model logic.
    This class contains common operations for all prediction UDFs:
        - accesses data part-by-part based on predefined batch size
        - manages the model cache
        - reads the corresponding model from BucketFS into cache
        - creates model pipeline through transformer api
        - manages the creation of predictions and the preparation of results.

    Additionally, the following
    methods should be implemented specifically for each UDF class:
        - create_dataframes_from_predictions
        - extract_unique_param_based_dataframes
        - execute_prediction
        - append_predictions_to_input_dataframe

    """
    def __init__(self,
                 exa,
                 batch_size: int,
                 pipeline: transformers.Pipeline,
                 base_model: ModelFactoryProtocol,
                 tokenizer: ModelFactoryProtocol,
                 task_type: str):
        self.exa = exa
        self.batch_size = batch_size
        self.pipeline = pipeline
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.device = None
        self.model_loader = None
        self.last_created_pipeline = None
        self.new_columns = []

    def run(self, ctx):
        device_id = ctx.get_dataframe(1).iloc[0]['device_id']
        self.device = device_management.get_torch_device(device_id)
        self.create_model_loader()
        ctx.reset()

        while True:
            batch_df = ctx.get_dataframe(num_rows=self.batch_size, start_col=1)
            if batch_df is None:
                break
            predictions_df = self.get_predictions_from_batch(batch_df)
            ctx.emit(predictions_df)

        self.model_loader.clear_device_memory()

    def create_model_loader(self):
        """
        Creates the model_loader.
        """
        self.model_loader = LoadLocalModel(pipeline_factory=self.pipeline,
                                           base_model_factory=self.base_model,
                                           tokenizer_factory=self.tokenizer,
                                           task_type=self.task_type,
                                           device=self.device)

    def get_predictions_from_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform separate predictions for each model in the dataframe.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Prediction results of the corresponding batched dataframe
        """
        result_df_list = []

        unique_model_dataframes = self.extract_unique_model_dataframes_from_batch(self, batch_df)
        for model_df in unique_model_dataframes:
            if "error_message" in model_df:
                result_df_list.append(model_df)
                continue
            try:
                self.check_cache(model_df)
            except Exception as exc:
                stack_trace = traceback.format_exc()
                result_with_error_df = self.get_result_with_error(
                    model_df, stack_trace)
                result_df_list.append(result_with_error_df)
            else:
                current_results_df_list = \
                    self.get_prediction_from_unique_param_based_dataframes(model_df)
                result_df_list.extend(current_results_df_list)

        result_df = pd.concat(result_df_list)
        return result_df.replace(np.nan, None)

    def get_prediction_from_unique_param_based_dataframes(self, model_df) \
            -> List[pd.DataFrame]:
        """
        Performs separate predictions for data with the same parameters
        in the same model dataframe.

        :param model_df: Dataframe containing data that has the same model
        but can have different parameters.

        :return: List of prediction results
        """
        result_df_list = []
        for param_based_model_df in self.extract_unique_param_based_dataframes(model_df):
            try:
                result_df = self.get_prediction(param_based_model_df)
                result_df_list.append(result_df)
            except Exception as exc:
                stack_trace = traceback.format_exc()
                result_with_error_df = self.get_result_with_error(
                    param_based_model_df, stack_trace)
                result_df_list.append(result_with_error_df)
        return result_df_list

    @staticmethod
    def _check_values_not_null(model_name, bucketfs_conn, sub_dir):
        if not (model_name and bucketfs_conn and sub_dir):
            error_message = f"For each model model_name, bucketfs_conn and sub_dir need to be provided. " \
                            f"Found model_name = {model_name}, bucketfs_conn = {bucketfs_conn}, sub_dir = {sub_dir}."
            raise ValueError(error_message)

    @staticmethod
    def extract_unique_model_dataframes_from_batch(self,
            batch_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique model dataframes with the same model_name, bucketfs_conn,
        and sub_dir from the dataframe.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Unique model dataframe having same model_name,
        bucketfs_connection, and sub_dir
        """

        unique_values = dataframe_operations.get_unique_values(
            batch_df, constants.ORDERED_COLUMNS, sort=True)

        for model_name, bucketfs_conn, sub_dir in unique_values:
            try:
                self._check_values_not_null(model_name, bucketfs_conn, sub_dir)
            except ValueError:
                stack_trace = traceback.format_exc()
                result_with_error_df = self.get_result_with_error(
                    batch_df, stack_trace)
                yield result_with_error_df
                return

            selections = ( #todo replace with specification in future?
                    (batch_df['model_name'] == model_name) &
                    (batch_df['bucketfs_conn'] == bucketfs_conn) &
                    (batch_df['sub_dir'] == sub_dir)
            )

            model_df = batch_df[
                selections]

            yield model_df

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
        current_model_specification = BucketFSModelSpecification(model_name, self.task_type, bucketfs_conn, sub_dir)

        if self.model_loader.current_model_specification != current_model_specification:
            bucketfs_location = \
                bucketfs_operations.create_bucketfs_location_from_conn_object(
                    self.exa.get_connection(bucketfs_conn))

            self.model_loader.clear_device_memory()
            self.model_loader.set_current_model_specification(current_model_specification)
            self.model_loader.set_bucketfs_model_cache_dir(bucketfs_location)

            try:
                self.last_created_pipeline = self.model_loader.load_models()
            except Exception as exc:
                stack_trace = traceback.format_exc()
                self.model_loader.last_model_loaded_successfully = False
                self.model_loader.model_load_error = stack_trace
                raise

        elif not self.model_loader.last_model_loaded_successfully:
            raise Exception("Model loading failed previously with :" + self.model_loader.model_load_error)

    def get_prediction(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform prediction of the given model and preparation of the prediction
        results according to the format that the UDF can emit.

        :param model_df: The dataframe to be predicted

        :return: The dataframe where the model_df is formatted with the
        prediction results
        """

        predictions = self.execute_prediction(model_df)
        pred_df_list = self.create_dataframes_from_predictions(predictions)
        pred_df = self.append_predictions_to_input_dataframe(
            model_df, pred_df_list)
        pred_df['error_message'] = None
        return pred_df

    def get_result_with_error(self, model_df: pd.DataFrame, stack_trace: str) \
            -> pd.DataFrame:
        """
        Add the stack trace to the dataframe that received an error
        during prediction.

        :param model_df: The dataframe that received an error during prediction
        :param stack_trace: String of the stack traceback
        """
        for col in self.new_columns:
            model_df[col] = None
        model_df["error_message"] = stack_trace
        return model_df

    @abstractmethod
    def create_dataframes_from_predictions(self, predictions: List[Any]) \
            -> List[pd.DataFrame]:
        pass

    @abstractmethod
    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        pass

    @abstractmethod
    def execute_prediction(
            self, model_df: pd.DataFrame) -> List[pd.DataFrame]:
        pass

    @abstractmethod
    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        pass
