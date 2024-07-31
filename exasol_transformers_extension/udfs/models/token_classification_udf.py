import pandas as pd
import transformers
from typing import List, Iterator, Any, Union, Dict
from exasol_transformers_extension.utils import dataframe_operations
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class TokenClassificationUDF(BaseModelUDF):
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModelForTokenClassification,
                 tokenizer=transformers.AutoTokenizer):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, task_type='token-classification')
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
        for aggregation_strategy in unique_params:
            current_strategy = aggregation_strategy[0]
            param_based_model_df = model_df[
                model_df['aggregation_strategy'] == current_strategy]

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

