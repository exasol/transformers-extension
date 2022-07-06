import torch
import tempfile
import transformers
from exasol_transformers_extension.udfs import bucketfs_operations


class SequenceClassification:
    def __init__(self, exa, base_model=transformers.AutoModel,
                 tokenizer=transformers.AutoTokenizer):
        self.exa = exa
        self.base_model = base_model
        self.tokenizer = tokenizer

    def run(self, ctx):
        bfs_conn: str = ctx[0]
        model_name: str = ctx[1]

        # set model path in buckets
        model_path = bucketfs_operations.get_model_path(model_name)

        # create bucketfs location
        bfs_conn_obj = self.exa.get_connection(bfs_conn)
        bucketfs_location = bucketfs_operations.create_bucketfs_location(
            bfs_conn_obj)

        with tempfile.TemporaryDirectory() as tmpdir_name:
            # download the model files from bucketfs
            bucketfs_operations.download_model_files_from_bucketfs(
                tmpdir_name, model_path, bucketfs_location)

            # load models
            model = self.base_model.from_pretrained(
                model_name, cache_dir=tmpdir_name)
            tokenizer = self.tokenizer.from_pretrained(
                model_name, cache_dir=tmpdir_name)

            for i in range(2, self.exa.meta.input_column_count):
                tokens = tokenizer(ctx[i], return_tensors="pt")
                logits = model(**tokens).logits
                predictions = torch.softmax(logits, dim=1).tolist()[0]
                ctx.emit(*predictions)


