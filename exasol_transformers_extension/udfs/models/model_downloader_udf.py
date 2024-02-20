from typing import Tuple

import transformers
from exasol_bucketfs_utils_python.bucketfs_factory import BucketFSFactory

from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory


class ModelDownloaderUDF:
    """
    UDF which downloads a pretrained model from Huggingface using Huggingface's transformers API,
    and uploads it to the BucketFS at , from where it can then be loaded without accessing Huggingface again.
    Must be called with the following Input Parameter:

    model_name                | sub_dir                 | bfs_conn            | token_conn
    ---------------------------------------------------------------------------------------------------
    name of Huggingface model | directory to save model | BucketFS connection | name of token connection

    returns <sub_dir/model_name> , <path of model BucketFS>
    """
    def __init__(self,
                 exa,
                 base_model_factory: ModelFactoryProtocol = transformers.AutoModel,
                 tokenizer_factory: ModelFactoryProtocol = transformers.AutoTokenizer,
                 huggingface_hub_bucketfs_model_transfer: HuggingFaceHubBucketFSModelTransferSPFactory =
                 HuggingFaceHubBucketFSModelTransferSPFactory(),
                 bucketfs_factory: BucketFSFactory = BucketFSFactory()):
        self._exa = exa
        self._base_model_factory = base_model_factory
        self._tokenizer_factory = tokenizer_factory
        self._huggingface_hub_bucketfs_model_transfer = huggingface_hub_bucketfs_model_transfer
        self._bucketfs_factory = bucketfs_factory

    def run(self, ctx) -> None:
        while True:
            model_path = self._download_model(ctx)
            ctx.emit(*model_path)
            if not ctx.next():
                break

    def _download_model(self, ctx) -> Tuple[str, str]:
        # parameters
        model_name = ctx.model_name     # name of Huggingface model
        sub_dir = ctx.sub_dir           # directory to save model
        bfs_conn = ctx.bfs_conn         # BucketFS connection
        token_conn = ctx.token_conn     # name of token connection

        # extract token from the connection if token connection name is given.
        # note that, token is required for private models. It doesn't matter
        # whether there is a token for public model or even what the token is.
        token = False
        if token_conn:
            token_conn_obj = self._exa.get_connection(token_conn)
            token = token_conn_obj.password

        # set model path in buckets
        model_path = bucketfs_operations.get_model_path(sub_dir, model_name)

        # create bucketfs location
        bfs_conn_obj = self._exa.get_connection(bfs_conn)
        bucketfs_location = self._bucketfs_factory.create_bucketfs_location(
            url=bfs_conn_obj.address,
            user=bfs_conn_obj.user,
            pwd=bfs_conn_obj.password
        )

        # download base model and tokenizer into the model path
        with self._huggingface_hub_bucketfs_model_transfer.create(
                bucketfs_location=bucketfs_location,
                model_name=model_name,
                model_path=model_path,
                token=token
        ) as downloader:
            for model in [self._base_model_factory, self._tokenizer_factory]:
                downloader.download_from_huggingface_hub(model)
            # upload model files to BucketFS
            model_tar_file_path = downloader.upload_to_bucketfs()

        return str(model_path), str(model_tar_file_path)
