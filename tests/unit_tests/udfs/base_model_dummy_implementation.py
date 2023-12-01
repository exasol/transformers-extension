
import pandas as pd
import transformers
from typing import List, Iterator, Any, Dict, Union
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class DummyModelLoader:
    """
    Create a Dummy model loader that does not create a transformers Pipeline,
     since that fails with test data.
    """
    def __init__(self,
                 base_model,
                 tokenizer,
                 task_name,
                 device
                 ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.device = device
        self.last_loaded_model = None
        self.last_loaded_tokenizer = None
        self.last_created_pipeline = None
        self.last_loaded_model_key = None

    def load_models(self, model_name: str,
                    current_model_key,
                    cache_dir,
                    token_conn_obj) -> None:
        token = False
        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=cache_dir, use_auth_token=token)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_auth_token=token)


class DummyImplementationUDF(BaseModelUDF):
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModel,
                 tokenizer=transformers.AutoTokenizer):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, 'dummy_task')
        self._desired_fields_in_prediction = ["answer", "score"]
        self.new_columns = ["answer", "score", "error_message"]

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        yield model_df

    def execute_prediction(self, model_df: pd.DataFrame) -> \
            List[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        dummy_result = [{"answer": True, "score": "1"}]
        return dummy_result

    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:

        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)
        return model_df

    def create_dataframes_from_predictions(
            self, predictions: List[Union[
                Dict[str, Any], List[Dict[str, Any]]]]) -> List[pd.DataFrame]:
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame(result, index=[0])
            results_df_list.append(result_df)
        return results_df_list

    def create_model_loader(self):
        """ overwrite the model loader creation with dummy model loader creation"""
        self.model_loader = DummyModelLoader(self.base_model,
                                             self.tokenizer,
                                             self.task_name,
                                             self.device)
