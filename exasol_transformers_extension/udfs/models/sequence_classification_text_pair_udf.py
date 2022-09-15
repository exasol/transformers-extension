import pandas as pd
import transformers
from typing import List
from exasol_transformers_extension.udfs.models.base_model_udf import \
    BaseModelUDF


class SequenceClassificationTextPairUDF(BaseModelUDF):
    def __init__(self,
                 exa,
                 batch_size=100,
                 pipeline=transformers.pipeline,
                 base_model=transformers.AutoModelForSequenceClassification,
                 tokenizer=transformers.AutoTokenizer):
        super().__init__(exa, batch_size, pipeline, base_model,
                         tokenizer, task_name='text-classification')

    def execute_prediction(self, model_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: List of dataframe includes prediction details
        """
        first_sequences = list(model_df['first_text'])
        second_sequences = list(model_df['second_text'])

        input_sequences = []
        for text, text_pair in zip(first_sequences, second_sequences):
            input_sequences.append({"text": text, "text_pair": text_pair})

        results = self.last_created_pipeline(
            input_sequences, return_all_scores=True)

        return self.create_dataframes_from_predictions(results)

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
        n_labels = list(map(lambda x: x.shape[0], pred_df_list))
        repeated_indexes = model_df.index.repeat(repeats=n_labels)
        model_df = model_df.loc[repeated_indexes].reset_index(drop=True)

        # Concat predictions and model_df
        pred_df = pd.concat(pred_df_list, axis=0).reset_index(drop=True)
        model_df = pd.concat([model_df, pred_df], axis=1)

        return model_df

