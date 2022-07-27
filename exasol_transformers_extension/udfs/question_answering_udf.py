import pandas as pd
import transformers
from typing import Tuple, List
from transformers import pipeline
from exasol_transformers_extension.udfs import bucketfs_operations


class QuestionAnswering:
    def __init__(self,
                 exa,
                 batch_size=100,
                 base_model=transformers.AutoModelForQuestionAnswering,
                 tokenizer=transformers.AutoTokenizer):
        self.exa = exa
        self.bacth_size = batch_size
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.cache_dir = None
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

        model_path = bucketfs_operations.get_model_path(sub_dir, model_name)
        self.cache_dir = bucketfs_operations.get_local_bucketfs_path(
            bucketfs_location=bucketfs_location, model_path=str(model_path))

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
        preds, labels = self._predict_model(model_df)
        pred_df = self._prepare_prediction_dataframe(model_df, preds, labels)
        return pred_df

    def _predict_model(self, model_df: pd.DataFrame) -> \
            Tuple[List[float], List[str]]:
        """
        Predict the given text list using recently loaded models, return
        probability scores and labels

        :param model_df: The dataframe to be predicted

        :return: A tuple containing prediction score list and label list
        """
        questions = list(model_df['question'])
        contexts = list(model_df['context_text'])
        question_answerer = pipeline("question-answering",
                                     model=self.last_loaded_model,
                                     tokenizer=self.last_loaded_tokenizer)
        results = question_answerer(question=questions, context=contexts)

        answers = []
        scores = []
        for result in results:
            answers.append(result['answer'])
            scores.append(result['score'])

        return scores, answers

    @staticmethod
    def _prepare_prediction_dataframe(
            model_df: pd.DataFrame, scores: List[float], answers: List[str]) \
            -> pd.DataFrame:
        """
        Reformat the dataframe used in prediction, such that each input rows
        has a row for each label and its probability score

        :param model_df: Dataframe used in prediction
        :param scores: List of prediction probabilities
        :param answers: List of predicted answers

        :return: Prepared dataframe including input data and predictions
        """
        model_df['answer'] = answers
        model_df['score'] = scores
        return model_df
