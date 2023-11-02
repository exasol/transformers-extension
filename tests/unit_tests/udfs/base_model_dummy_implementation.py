
import pandas as pd
import transformers
from typing import List, Iterator, Any, Dict, Union
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


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

    def load_models(self, model_name: str, token_conn_name: str) -> None:
        token = False
        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=self.cache_dir, use_auth_token=token)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir, use_auth_token=token)