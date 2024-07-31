import pandas as pd
import transformers
from typing import List, Any, Iterator, Dict
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class TextGenerationUDF(BaseModelUDF):
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModelForCausalLM,
                 tokenizer=transformers.AutoTokenizer):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, task_type='text-generation')
        self.new_columns = ["generated_text", "error_message"]

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same max_length and return_full_text
        parameter values

        :param model_df: Dataframe used in prediction

         :return: Unique model dataframes having same specified parameters
        """

        unique_params = dataframe_operations.get_unique_values(
            model_df, ['max_length', 'return_full_text'])
        for max_length, return_full_text in unique_params:
            param_based_model_df = model_df[
                (model_df['max_length'] == max_length) &
                (model_df['return_full_text'] == return_full_text)]

            yield param_based_model_df

    def execute_prediction(self, model_df: pd.DataFrame) \
            -> List[Dict[str, Any]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: A tuple containing prediction score list and label list
        """
        text_data = list(model_df['text_data'])
        max_length = int(model_df['max_length'].iloc[0])
        return_full_text = bool(model_df['return_full_text'].iloc[0])
        results = self.last_created_pipeline(
            text_data, max_length=max_length, return_full_text=return_full_text)

        #  Batch prediction returns list of list while single prediction just
        #  return a list. In case of batch predictions, we need to flatten
        #  2D prediction results to 1D list
        results = sum(results, []) if type(results[0]) == list else results
        return results

    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a generated text

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """
        pred_df = pd.concat(pred_df_list, axis=0) \
            .reset_index(drop=True)
        model_df = pd.concat([
            model_df.reset_index(drop=True),
            pred_df
        ], axis=1)

        return model_df

    def create_dataframes_from_predictions(
            self, predictions:  List[Dict[str, Any]]) -> List[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: predictions results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            results_df_list.append(
                pd.DataFrame(
                    data=[result['generated_text']],
                    columns=['generated_text'])
            )
        return results_df_list
