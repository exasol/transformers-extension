from typing import Tuple, List

import pandas as pd
import torch
import transformers
from exasol_transformers_extension.udfs import bucketfs_operations


class SequenceClassificationSingleText:
    def __init__(self,
                 exa,
                 cache_dir=None,
                 batch_size=100,
                 base_model=transformers.AutoModelForSequenceClassification,
                 tokenizer=transformers.AutoTokenizer):
        self.exa = exa
        self.cache_dir = cache_dir
        self.bacth_size = batch_size
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.last_loaded_model_name = None
        self.last_loaded_model = None
        self.last_loaded_tokenizer = None

    def run(self, ctx):
        while True:
            batch_df = ctx.get_dataframe(self.bacth_size)
            if batch_df is None:
                break

            result_df = self.get_batched_predictions(batch_df)
            ctx.emit(result_df)

    def get_batched_predictions(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform separate predictions for each model in the dataframe. If the
        model is not cached, it is loaded into the cache before the prediction.

        :param batch_df: A batch of dataframe retrieved from context

        :return: Prediction results of the corresponding dataframe
        """
        result_df_list = []
        for model_name in batch_df['model_name'].unique():
            model_df = batch_df[batch_df['model_name'] == model_name]

            if self.last_loaded_model_name != model_name:
                self.set_cache_dir(model_df)
                self.load_models(model_name)

            model_pred_df = self.get_prediction(model_df)
            result_df_list.append(model_pred_df)

        result_df = pd.concat(result_df_list)
        return result_df

    def set_cache_dir(self, model_df: pd.DataFrame) -> None:
        """
        Set the cache directory in bucketfs of the specified model. Note that,
        cache_dir class variable is used for testing purpose. This variable is
        set to a local path only in unit tests.

        :param model_df: The model dataframe to set the cache directory
        """
        model_name = model_df['model_name'].iloc[0]
        bucketfs_conn_name = model_df['bucketfs_conn'].iloc[0]
        sub_dir = model_df['sub_dir'].iloc[0]
        bucketfs_location = bucketfs_operations.create_bucketfs_location(
            self.exa.get_connection(bucketfs_conn_name))

        if not self.cache_dir:
            model_path = bucketfs_operations.get_model_path(sub_dir, model_name)
            self.cache_dir = bucketfs_operations.get_local_bucketfs_path(
                bucketfs_location=bucketfs_location,
                model_path=f"container/{model_path}")

    def load_models(self, model_name: str) -> None:
        """
        Load model and tokenizer model from the cached location in bucketfs

        :param model_name: The model name to be loaded
        """
        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_model_name = model_name

    def get_prediction(self, model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform prediction of the given model and preparation of the prediction
        results according to the format that the UDF can emit.

        :param model_df: The dataframe to be predicted

        :return: The dataframe where the model_df is formatted with the
        prediction results
        """
        preds, labels = self._predict_model(list(model_df['text_data']))
        pred_df = self._prepare_prediction_dataframe(model_df, preds, labels)
        return pred_df

    def _predict_model(self, sequences: List[str]) -> \
            Tuple[List[float], List[str]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param sequences: The list of text to be predicted

        :return: A tuple containing prediction score list and label list
        """
        tokens = self.last_loaded_tokenizer(sequences, return_tensors="pt")
        logits = self.last_loaded_model(**tokens).logits
        preds = torch.softmax(logits, dim=1).tolist()
        labels_dict = self.last_loaded_model.config.id2label
        labels = list(map(lambda x: x[1], sorted(labels_dict.items())))

        return preds, labels

    @staticmethod
    def _prepare_prediction_dataframe(
            model_df: pd.DataFrame, preds: List[float], labels: List[str]) \
            -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score

        :param model_df: Dataframe used in prediction
        :param preds: List of prediction probabilities
        :param labels: List of labels

        :return: Prepared dataframe including input data and predictions
        """
        n_labels = len(labels)

        # Repeat each row consecutively as the number of labels. At the end,
        # the dataframe is expanded from (m, n) to (m*n_labels, n)
        repeated_indexes = model_df.index.repeat(n_labels)
        model_df = model_df.loc[repeated_indexes].reset_index(drop=True)

        # Fill the dataframe with labels repeatedly, such that each input row
        # has a row for each label
        extension_factor = model_df.shape[0]//n_labels
        model_df['label'] = labels * extension_factor

        # Flatten 2D prediction scores to 1D list and assign it to score
        # column of the dataframe. We use for this the sum function with a list as inital value 
        # and + operator of lists
        preds_flatten = sum(preds, [])
        model_df['score'] = [round(pred, 2) for pred in preds_flatten]

        return model_df
