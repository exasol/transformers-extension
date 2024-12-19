
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
                 tokenizer=transformers.AutoTokenizer,
                 work_with_spans: bool = False
                 ):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, task_type='dummy_task',
                         work_with_spans=work_with_spans)
        self._desired_fields_in_prediction = ["answer", "score"]
        self.new_columns = ["answer", "score", "error_message"]

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        yield model_df

    def execute_prediction(self, model_df: pd.DataFrame) -> \
            List[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        input_data = list(model_df['input_data'])
        results = self.last_created_pipeline(input_data)
        return results

    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        print("'''''''''''''''''''''''")
        print(pred_df_list)
        model_df = model_df.reset_index(drop=True)
        print("'''''''''''''''''''''''")
        print(model_df)
        print("'''''''''''''''''''''''")

        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)
        print("'''''''''''''''''''''''")
        print(model_df)
        print("'''''''''''''''''''''''")
        if self.work_with_spans:
            model_df = self.create_new_span_columns(model_df)
            model_df = self.drop_old_data_for_span_execution(model_df)
        return model_df

    def create_dataframes_from_predictions(
            self, predictions: List[Union[
                Dict[str, Any], List[Dict[str, Any]]]]) -> List[pd.DataFrame]:
        results_df_list = []
        for result in predictions:
            print(result)
            result_df = pd.DataFrame(result)#, index=[0])
            results_df_list.append(result_df)
            print("____")
        print(results_df_list)
        return results_df_list

    def create_new_span_columns(self, model_df: pd.DataFrame) \
            -> pd.DataFrame: #todo add back span test cases for all tests
        model_df[["test_span_column_add"]] = 'add_this',
        return model_df#todo or do we not do these, and keep all test cases but
        #todo only for with spans for the relevant udfs?

    def drop_old_data_for_span_execution(self, model_df: pd.DataFrame)\
            -> pd.DataFrame:
        model_df = model_df.drop(columns=["test_span_column_drop"])
        return model_df
