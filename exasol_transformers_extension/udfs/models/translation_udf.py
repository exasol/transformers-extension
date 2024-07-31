import pandas as pd
import transformers
from typing import List, Iterator, Any, Optional, Dict
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class TranslationUDF(BaseModelUDF):
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModelForSeq2SeqLM,
                 tokenizer=transformers.AutoTokenizer):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, task_type='translation')
        self._translation_prefix = "translate {src_lang} to {target_lang}: "
        self.new_columns = ["translation_text", "error_message"]

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same max_length, source_language,
        and target_language parameter values

        :param model_df: Dataframe used in prediction

         :return: Unique model dataframes having same specified parameters
        """

        unique_params = dataframe_operations.get_unique_values(
            model_df, ['max_length', 'source_language', 'target_language'])
        for max_length, source_language, target_language in unique_params:
            param_based_model_df = model_df[
                (model_df['max_length'] == max_length) &
                (model_df['source_language'] == source_language) &
                (model_df['target_language'] == target_language)]

            yield param_based_model_df

    def execute_prediction(self, model_df: pd.DataFrame) \
            -> List[Dict[str, Any]]:
        """
        Predict the given text list using recently loaded models, return
        translated text

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        source_language = str(model_df['source_language'].iloc[0])
        target_language = str(model_df['target_language'].iloc[0])
        translation_prefix = ''
        if source_language and target_language:
            translation_prefix = self._translation_prefix.format(
                src_lang=source_language, target_lang=target_language)

        text_data = list(translation_prefix + model_df['text_data'].astype(str))
        max_length = int(model_df['max_length'].iloc[0])

        results = self.last_created_pipeline(text_data, max_length=max_length)
        return results

    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input row
        has a row for each translated texts

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """
        pred_df = pd.concat(pred_df_list, axis=0)\
            .reset_index(drop=True)
        model_df = pd.concat([
            model_df.reset_index(drop=True),
            pred_df
        ], axis=1)

        return model_df

    def create_dataframes_from_predictions(
            self, predictions: List[Dict[str, Any]]) -> List[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: predictions results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame([result])
            results_df_list.append(result_df)

        return results_df_list
