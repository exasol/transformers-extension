import pandas as pd
import transformers
from typing import List, Iterator, Any, Dict, Union
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class QuestionAnsweringUDF(BaseModelUDF):
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModelForQuestionAnswering,
                 tokenizer=transformers.AutoTokenizer):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, 'question-answering')
        self._desired_fields_in_prediction = ["answer", "score"]
        self.new_columns = ["answer", "score", "rank", "error_message"]

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same top_k parameter values

        :param model_df: Dataframe used in prediction

         :return: Unique model dataframes having specified parameters
        """
        unique_params = dataframe_operations. \
            get_unique_values(model_df, ['top_k'])
        for top_k in unique_params:
            current_top_k = top_k[0]
            param_based_model_df = model_df[
                model_df['top_k'] == current_top_k]

            yield param_based_model_df

    def execute_prediction(self, model_df: pd.DataFrame) -> \
            List[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframes holding prediction results
        """
        questions = list(model_df['question'])
        contexts = list(model_df['context_text'])
        top_k = int(model_df['top_k'].iloc[0])
        results = self.last_created_pipeline(
            question=questions, context=contexts, top_k=top_k)

        # We need to separate the answer to one question from the answers to
        # multiple questions, such that results of one question could be
        # - a dict where top_k=1, or
        # - either a dict or list of dicts where top_k>1
        # in both cases we need to put the answer(s) in a list to make sure that
        # the answer(s) is from a single question
        results = [results] if len(questions) == 1 else results
        return results

    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score

        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of dataframes holding prediction results

        :return: Prepared dataframe including input data and predictions
        """
        n_topk_results = list(map(lambda x: x.shape[0], pred_df_list))
        repeated_indexes = model_df.index.repeat(repeats=n_topk_results)
        model_df = model_df.loc[repeated_indexes].reset_index(drop=True)

        # Concat predictions and model_df
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)

        return model_df

    def create_dataframes_from_predictions(
            self, predictions: List[Union[
                Dict[str, Any], List[Dict[str, Any]]]]) -> List[pd.DataFrame]:
        """
        Convert predictions to dataframe.

        :param predictions: prediction results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame([result]) if type(result) == dict \
                else pd.DataFrame(result)
            result_df = result_df[self._desired_fields_in_prediction]
            result_df["rank"] = result_df["score"].rank(
                ascending=False, method='dense').astype(int)
            results_df_list.append(result_df)

        return results_df_list
