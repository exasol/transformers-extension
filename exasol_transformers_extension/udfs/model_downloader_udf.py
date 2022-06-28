import os
import tempfile
import transformers
from pathlib import PurePosixPath, Path
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation


class ModelDownloader:
    def __init__(self, exa, base_model_downloader=transformers.AutoModel,
                 tokenizer_downloader=transformers.AutoTokenizer):
        self.exa = exa
        self.base_model_downloader = base_model_downloader
        self.tokenizer_downloader = tokenizer_downloader

    def run(self, ctx) -> None:
        model_name = ctx.model_name
        bfs_conn = ctx.bfs_conn

        # set model path in buckets
        model_path = model_name.replace('-', '_')

        # create bucketfs location
        bfs_conn_obj = self.exa.get_connection(bfs_conn)
        bucketfs_location = _create_bucketfs_location(bfs_conn_obj)

        # download base model and tokenizer into the model path
        for downloader in \
                [self.base_model_downloader, self.tokenizer_downloader]:

            with tempfile.TemporaryDirectory() as tmpdirname:
                # download model into tmp folder
                downloader.from_pretrained(model_name, cache_dir=tmpdirname)

                # upload the downloaded model files into bucketfs
                _upload_model_file_to_bucketfs(
                    tmpdirname, model_path, bucketfs_location)

        ctx.emit(model_path)


def _create_bucketfs_location(bfs_conn_obj) -> BucketFSLocation:
    return BucketFSFactory().create_bucketfs_location(
        url=bfs_conn_obj.address,
        user=bfs_conn_obj.user,
        pwd=bfs_conn_obj.password)


def _upload_model_file_to_bucketfs(
        tmpdirname: str, model_path:str,  bucketfs_location: BucketFSLocation):
    for tmp_file_path in Path(tmpdirname).iterdir():
        with open(tmp_file_path, mode='rb') as file:
            bucketfs_path = PurePosixPath(
                model_path, tmp_file_path.relative_to(tmpdirname))
            bucketfs_location.upload_fileobj_to_bucketfs(
                file, str(bucketfs_path))
