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

            if self.last_loaded_model_name != model_name:
                bucketfs_conn = model_df['bucketfs_conn'].iloc[0]
                sub_dir = model_df['sub_dir'].iloc[0]
                self._set_cache_dir(sub_dir, model_name, bucketfs_conn)
                self.load_models(model_name)

            model_pred_df = self.model_prediction(model_df)
            result_df_list.append(model_pred_df)

        result_df = pd.concat(result_df_list)
        return result_df

    def load_models(self, model_name):
        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir)
        self.last_loaded_model_name = model_name

    def model_prediction(self, model_df):

        tokens = self.last_loaded_tokenizer(
            list(model_df['text_data']), return_tensors="pt")
        logits = self.last_loaded_model(**tokens).logits
        preds = torch.softmax(logits, dim=1).tolist()
        labels_dict = self.last_loaded_model.config.id2label
        labels = list(map(lambda x: x[1], sorted(labels_dict.items())))

        model_df = model_df.loc[
            model_df.index.repeat(len(labels))].reset_index(drop=True)
        model_df['label'] = labels * (model_df.shape[0]//len(labels))
        model_df['score'] = sum(preds, [])

        return model_df

    def _get_bucketfs_location(self, bucketfs_conn: str):
        bucketfs_conn = self.exa.get_connection(bucketfs_conn)
        return bucketfs_operations.create_bucketfs_location(bucketfs_conn)

    def _set_cache_dir(self, sub_dir, model_name, bucketfs_conn_name):
        bucketfs_location = self._get_bucketfs_location(bucketfs_conn_name)
        if not self.cache_dir:
            model_path = bucketfs_operations.get_model_path(sub_dir, model_name)
            self.cache_dir = bucketfs_operations.get_local_bucketfs_path(
                bucketfs_location=bucketfs_location,
                model_path=f"container/{model_path}")
