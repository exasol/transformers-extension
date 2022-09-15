from abc import abstractmethod
from typing import Iterator, List, Any, Optional
import torch
import pandas as pd
from exasol_transformers_extension.deployment import constants
from exasol_transformers_extension.utils import device_management, \
    bucketfs_operations, dataframe_operations


class BaseModelUDF:
    def __init__(self,
                 exa,
                 batch_size,
                 pipeline,
                 base_model,
                 tokenizer,
                 task_name):
        self.exa = exa
        self.batch_size = batch_size
        self.pipeline = pipeline
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.device = None
        self.cache_dir = None
        self.last_loaded_model_key = None
        self.last_loaded_model = None
        self.last_loaded_tokenizer = None
        self.last_created_pipeline = None

    def run(self, ctx):
        device_id = ctx.get_dataframe(1).iloc[0]['device_id']
        self.device = device_management.get_torch_device(device_id)
        ctx.reset()

        while True:
            batch_df = ctx.get_dataframe(num_rows=self.batch_size, start_col=1)
            if batch_df is None:
                break

            predictions_df = self.get_predictions_from_batch(batch_df)
            ctx.emit(predictions_df)

        self.clear_device_memory()

    def get_predictions_from_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform separate predictions for each model in the dataframe.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Prediction results of the corresponding batched dataframe
        """
        result_df_list = []
        for model_df in \
                self.extract_unique_model_dataframes_from_batch(batch_df):
            for param_based_model_df in \
                    self.extract_unique_param_based_dataframes(model_df):
                result_df = self.get_prediction(param_based_model_df)
                result_df_list.append(result_df)

        result_df = pd.concat(result_df_list)
        return result_df

    def extract_unique_model_dataframes_from_batch(
            self, batch_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique model dataframes with the same model_name, bucketfs_conn,
        and sub_dir from the dataframe. If the extracted model is not cached,
        it is loaded into the cache before performing the prediction.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Unique model dataframes having same model_name,
        bucketfs_connection, and sub_dir
        """

        unique_values = dataframe_operations.get_unique_values(
            batch_df, constants.ORDERED_COLUMNS, sort=True)
        for model_name, bucketfs_conn, sub_dir in unique_values:
            model_df = batch_df[
                (batch_df['model_name'] == model_name) &
                (batch_df['bucketfs_conn'] == bucketfs_conn) &
                (batch_df['sub_dir'] == sub_dir)]

            current_model_key = (bucketfs_conn, sub_dir, model_name)
            if self.last_loaded_model_key != current_model_key:
                self.set_cache_dir(model_df)
                self.clear_device_memory()
                self.load_models(model_name)
                self.last_loaded_model_key = current_model_key

            yield model_df

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same model parameter values. if there
        is no model specified parameter, the input dataframe return as it is.

        :param model_df: Dataframe used in prediction

        :return: Unique model dataframes having specified parameters
        """

        yield model_df

    def set_cache_dir(self, model_df: pd.DataFrame) -> None:
        """
        Set the cache directory in bucketfs of the specified model.

        :param model_df: The model dataframe to set the cache directory
        """
        model_name = model_df['model_name'].iloc[0]
        bucketfs_conn_name = model_df['bucketfs_conn'].iloc[0]
        sub_dir = model_df['sub_dir'].iloc[0]
        bucketfs_location = bucketfs_operations.create_bucketfs_location(
            self.exa.get_connection(bucketfs_conn_name))

        model_path = bucketfs_operations.get_model_path(sub_dir, model_name)
        self.cache_dir = bucketfs_operations.get_local_bucketfs_path(
            bucketfs_location=bucketfs_location, model_path=str(model_path))

    def clear_device_memory(self):
        """
        Delete models and free device memory
        """

        del self.last_loaded_model
        del self.last_loaded_tokenizer
        torch.cuda.empty_cache()

    def load_models(self, model_name: str, **kwargs) -> None:
        """
        Load model and tokenizer model from the cached location in bucketfs

        :param model_name: The model name to be loaded
        """
        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_created_pipeline = self.pipeline(
            self.task_name,
            model=self.last_loaded_model,
            tokenizer=self.last_loaded_tokenizer,
            device=self.device,
            framework="pt")

    def get_prediction(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform prediction of the given model and preparation of the prediction
        results according to the format that the UDF can emit.

        :param model_df: The dataframe to be predicted

        :return: The dataframe where the model_df is formatted with the
        prediction results
        """
        pred_df_list = self.execute_prediction(model_df)
        pred_df = self.append_predictions_to_input_dataframe(
            model_df, pred_df_list)
        return pred_df

    @staticmethod
    def create_dataframes_from_predictions(
            results: List[Any], columns: Optional[List[str]] = None) \
            -> List[pd.DataFrame]:
        """
        Convert predictions to dataframe. If the prediction results can be
        presented as is, the results are converted directly into the dataframe.
        Otherwise, model-specific adjustments must be made in each model's
        own class.

        :param results: Predictions results
        :param columns: Used columns in prediction

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in results:
            result_df = pd.DataFrame(result)
            results_df_list.append(result_df)

        return results_df_list

    @abstractmethod
    def execute_prediction(
            self, model_df: pd.DataFrame) -> List[pd.DataFrame]:
        pass

    @abstractmethod
    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        pass
