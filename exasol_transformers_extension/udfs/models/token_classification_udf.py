import pandas as pd
import transformers
from typing import List, Iterator, Any, Union, Dict
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class TokenClassificationUDF(BaseModelUDF):
    """
    UDF for finding and classifying a token/entity in a given text.
    If given an input span, text_data_char_begin and text_data_char_end should
    represent the entire input text and not indicate a substring.
    """
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModelForTokenClassification,
                 tokenizer=transformers.AutoTokenizer,
                 work_with_spans: bool = False
                 ):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, task_type='token-classification',
                         work_with_spans=work_with_spans)
        self._default_aggregation_strategy = 'simple'
        self._desired_fields_in_prediction = [
            "start", "end", "word", "entity", "score"]
        self.new_columns = [
            "start_pos", "end_pos", "word", "entity", "score", "error_message"]

    def extract_unique_param_based_dataframes(
            self, model_df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        Extract unique dataframes having same aggregation_strategy
        parameter values

        :param model_df: Dataframe used in prediction

         :return: Unique model dataframes having same specified parameters
        """
        model_df['aggregation_strategy'] = \
            model_df['aggregation_strategy'].fillna(
                self._default_aggregation_strategy)

        unique_params = dataframe_operations.get_unique_values(
            model_df, ['aggregation_strategy'])
        for unique_param in unique_params:
            current_aggregation_strategy = unique_param[0]
            param_based_model_df = model_df[
                model_df['aggregation_strategy'] == current_aggregation_strategy]

            yield param_based_model_df

    def execute_prediction(self, model_df: pd.DataFrame) -> List[Union[
                Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores, entities and associated words

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        text_data = list(model_df['text_data'])
        aggregation_strategy = model_df['aggregation_strategy'].iloc[0]
        results = self.last_created_pipeline(
            text_data, aggregation_strategy=aggregation_strategy)
        results = results if type(results[0]) == list else [results]

        if aggregation_strategy == "none":
            self._desired_fields_in_prediction = [
                "start", "end", "word", "entity", "score"]
        else:
            self._desired_fields_in_prediction = [
                "start", "end", "word", "entity_group", "score"]

        return results

    def create_new_span_columns(self, model_df: pd.DataFrame) -> pd.DataFrame:
        # create new columns for use with spans
        model_df[["entity_docid", "entity_char_begin", "entity_char_end"]] = None, None, None
        # we use different names in udf with span and without, so need to rename
        # this decision was made as to improve the naming of the columns without
        # breaking the interface of the existing udf
        model_df = model_df.rename(
            columns={
                "word": "entity_covered_text",
                "entity": "entity_type"})
        return model_df

    def drop_old_data_for_span_execution(self, model_df: pd.DataFrame) -> pd.DataFrame:
        # drop columns which are made superfluous by the spans to save data transfer
        model_df = model_df.drop(columns=["text_data", "start_pos", "end_pos"])
        return model_df

    def make_entity_span(self, df_row):
        token_docid = df_row["text_data_docid"]
        token_char_begin = df_row["start_pos"] + df_row['text_data_char_begin']
        token_char_end = df_row["end_pos"] + df_row['text_data_char_begin']
        return pd.Series([token_docid, token_char_begin, token_char_end])

    def append_predictions_to_input_dataframe(
            self, model_df: pd.DataFrame, pred_df_list: List[pd.DataFrame]) \
            -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score
        :param model_df: Dataframe used in prediction
        :param pred_df_list: List of predictions dataframes

        :return: Prepared dataframe including input data and predictions
        """

        # Repeat each row consecutively as the number of entities. At the end,
        # the dataframe is expanded from (m, n) to (m*n_entities, n)
        n_entities = list(map(lambda x: x.shape[0], pred_df_list))
        repeated_indexes = model_df.index.repeat(repeats=n_entities)
        model_df = model_df.loc[repeated_indexes].reset_index(drop=True)

        # Concat predictions and model_df
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)
        if self.work_with_spans:
            model_df = self.create_new_span_columns(model_df)
            model_df[["entity_docid", "entity_char_begin", "entity_char_end"]] =\
                model_df.apply(self.make_entity_span, axis=1)
            model_df = self.drop_old_data_for_span_execution(model_df)
        return model_df

    def create_dataframes_from_predictions(
            self, predictions:  List[Union[
                Dict[str, Any], List[Dict[str, Any]]]]) -> List[pd.DataFrame]:
        """
        Convert predictions to dataframe. Only score and answer fields are
        presented.

        :param predictions: predictions results

        :return: List of prediction dataframes
        """
        results_df_list = []
        for result in predictions:
            result_df = pd.DataFrame(result)
            result_df = result_df[self._desired_fields_in_prediction].rename(
                columns={
                    "start": "start_pos",
                    "end": "end_pos",
                    "entity_group": "entity"})
            results_df_list.append(result_df)

        return results_df_list

