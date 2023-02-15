import tempfile
import transformers
from exasol_transformers_extension.utils import bucketfs_operations


class ModelDownloader:
    def __init__(self, exa, base_model_downloader=transformers.AutoModel,
                 tokenizer_downloader=transformers.AutoTokenizer):
        self.exa = exa
        self.base_model_downloader = base_model_downloader
        self.tokenizer_downloader = tokenizer_downloader

    def run(self, ctx) -> None:
        while True:
            model_path = self._download_model(ctx)
            ctx.emit(model_path)
            if not ctx.next():
                break

    def _download_model(self, ctx) -> str:
        # parameters
        model_name = ctx.model_name
        sub_dir = ctx.sub_dir
        bfs_conn = ctx.bfs_conn

        # set model path in buckets
        model_path = bucketfs_operations.get_model_path(sub_dir, model_name)

        # create bucketfs location
        bfs_conn_obj = self.exa.get_connection(bfs_conn)
        bucketfs_location = \
            bucketfs_operations.create_bucketfs_location_from_conn_object(
                bfs_conn_obj)

        # download base model and tokenizer into the model path
        for downloader in \
                [self.base_model_downloader, self.tokenizer_downloader]:
            with tempfile.TemporaryDirectory() as tmpdir_name:
                # download model into tmp folder
                downloader.from_pretrained(model_name, cache_dir=tmpdir_name)

                # upload the downloaded model files into bucketfs
                bucketfs_operations.upload_model_files_to_bucketfs(
                    tmpdir_name, model_path, bucketfs_location)

        return str(model_path)




