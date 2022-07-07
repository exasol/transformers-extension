import torch
import transformers
from exasol_transformers_extension.udfs import bucketfs_operations


class SequenceClassificationSingleText:
    def __init__(self, exa, base_model=transformers.AutoModel,
                 tokenizer=transformers.AutoTokenizer):
        self.exa = exa
        self.base_model = base_model
        self.tokenizer = tokenizer

    def run(self, ctx):
        bucketfs_conn = ctx.bucketfs_conn
        model_name = ctx.model_name
        text = ctx.text

        # set model path in buckets
        model_path = bucketfs_operations.get_model_path(model_name)

        # get cache directory
        bfs_conn_obj = self.exa.get_connection(bucketfs_conn)
        bucketfs_location = \
            bucketfs_operations.create_bucketfs_location(bfs_conn_obj)
        cache_dir = bucketfs_operations.get_local_bucketfs_path(
            bucketfs_location, model_path)

        # load models
        model = self.base_model.from_pretrained(
            model_name, cache_dir=cache_dir)
        tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)

        # perform prediction
        tokens = tokenizer(text, return_tensors="pt")
        logits = model(**tokens).logits
        predictions = torch.softmax(logits, dim=1).tolist()[0]
        ctx.emit(*predictions)
