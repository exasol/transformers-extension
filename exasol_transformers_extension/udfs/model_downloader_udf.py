import os
import tempfile
import transformers
from pathlib import PurePosixPath
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory


class ModelDownloader:
    def __init__(self, exa, downloader_method=transformers.AutoModel):
        self.exa = exa
        self.downloader_method = downloader_method

    def run(self, ctx) -> None:
        model_name = ctx.model_name
        bfs_conn = ctx.bfs_conn

        # set model path in buckets
        model_path = model_name.replace('-', '_')

        # create bucketfs location
        bfs_conn_obj = self.exa.get_connection(bfs_conn)
        bucketfs_location = BucketFSFactory().create_bucketfs_location(
            url=bfs_conn_obj.address,
            user=bfs_conn_obj.user,
            pwd=bfs_conn_obj.password)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # download model into tmp folder
            self.downloader_method.from_pretrained(
                model_name, cache_dir=tmpdirname)

            # upload the downloaded model files into bucketfs
            for tmp_file_name in os.listdir(tmpdirname):
                tmp_file_path = os.path.join(tmpdirname, tmp_file_name)
                with open(tmp_file_path, mode='rb') as file:
                    bucketfs_path = PurePosixPath(model_path, tmp_file_name)
                    bucketfs_location.upload_fileobj_to_bucketfs(
                        file, str(bucketfs_path))

        ctx.emit(model_path)
