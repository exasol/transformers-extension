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

    def get_batched_predictions(self, batch_df):
        result_df_list = []
        for model_name in batch_df['model_name'].unique():
            model_df = batch_df[batch_df['model_name'] == model_name]
            bucketfs_conn = model_df['bucketfs_conn'].iloc[0]

            if self.last_loaded_model_name != model_name:
                self.load_models(model_name, bucketfs_conn)

            model_pred_df = self.model_prediction(model_df)
            result_df_list.append(model_pred_df)

        result_df = pd.concat(result_df_list)
        return result_df

    def load_models(self, model_name, bucketfs_conn):
        self._set_cache_directory(model_name, bucketfs_conn)

        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_model_name = model_name

    def model_prediction(self, model_df):
        bucketfs_conn = model_df['bucketfs_conn'].iloc[0]
        model_name = model_df['model_name'].iloc[0]

        model_preds = []
        for ix, row in model_df.iterrows():
            text_data = row['text_data']
            tokens = self.last_loaded_tokenizer(text_data, return_tensors="pt")
            logits = self.last_loaded_model(**tokens).logits
            preds = torch.softmax(logits, dim=1).tolist()[0]
            labels = self.last_loaded_model.config.id2label
            for i in range(len(preds)):
                model_preds.append([bucketfs_conn, model_name,
                                    text_data, labels[i], preds[i]])

        model_pred_df = pd.DataFrame(data=model_preds,
                                     columns=['bucketfs_conn', 'model_name',
                                              'text_data', 'label', 'score'])
        return model_pred_df

    def _get_bucketfs_location(self, bucketfs_conn: str):
        bucketfs_conn = self.exa.get_connection(bucketfs_conn)
        return bucketfs_operations.create_bucketfs_location(bucketfs_conn)

    def _set_cache_directory(self, model_name, bucketfs_conn_name):
        bucketfs_location = self._get_bucketfs_location(bucketfs_conn_name)
        if not self.cache_dir:
            model_path = bucketfs_operations.get_model_path(model_name)
            self.cache_dir = bucketfs_operations.get_local_bucketfs_path(
                bucketfs_location=bucketfs_location,
                model_path=f"container/{model_path}")