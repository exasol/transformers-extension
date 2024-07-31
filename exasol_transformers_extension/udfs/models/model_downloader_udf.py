from typing import Tuple

import transformers

from exasol_transformers_extension.utils import bucketfs_operations
from exasol_transformers_extension.utils.bucketfs_model_specification import \
    BucketFSModelSpecificationFactory
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory


class ModelDownloaderUDF:
    """
    UDF which downloads a pretrained model from Huggingface using Huggingface's transformers API,
    and uploads it to the BucketFS, from where it can then be loaded without accessing Huggingface again.
    Must be called with the following Input Parameter:

    model_name                | sub_dir                 | bfs_conn            | token_conn
    ---------------------------------------------------------------------------------------------------
    name of Huggingface model | directory to save model | BucketFS connection | name of token connection

    returns <sub_dir/model_name> , <path of model BucketFS>
    """
    def __init__(self,
                 exa,
                 tokenizer_factory: ModelFactoryProtocol = transformers.AutoTokenizer,
                 huggingface_hub_bucketfs_model_transfer: HuggingFaceHubBucketFSModelTransferSPFactory =
                 HuggingFaceHubBucketFSModelTransferSPFactory(),
                 current_model_specification_factory: BucketFSModelSpecificationFactory = BucketFSModelSpecificationFactory()):
        self._exa = exa
        self._tokenizer_factory = tokenizer_factory
        self._huggingface_hub_bucketfs_model_transfer = huggingface_hub_bucketfs_model_transfer
        self._current_model_specification_factory = current_model_specification_factory

    def run(self, ctx) -> None:
        while True:
            model_path = self._download_model(ctx)
            ctx.emit(*model_path)
            if not ctx.next():
                break

    def _download_model(self, ctx) -> Tuple[str, str]:
        # parameters
        bfs_conn = ctx.bfs_conn         # BucketFS connection
        token_conn = ctx.token_conn     # name of token connection
        current_model_specification = self._current_model_specification_factory.create(ctx.model_name,
                                                                                       ctx.task_type,
                                                                                       bfs_conn,
                                                                                       ctx.sub_dir)   # specifies details of Huggingface model

        model_factory = current_model_specification.get_model_factory()
        # extract token from the connection if token connection name is given.
        # note that, token is required for private models. It doesn't matter
        # whether there is a token for public model or even what the token is.
        token = False
        if token_conn:
            token_conn_obj = self._exa.get_connection(token_conn)
            token = token_conn_obj.password

        # set model path in buckets
        model_path = current_model_specification.get_bucketfs_model_save_path()

        # create bucketfs location
        bfs_conn_obj = self._exa.get_connection(bfs_conn)
        bucketfs_location = bucketfs_operations.create_bucketfs_location_from_conn_object(bfs_conn_obj)

        # download base model and tokenizer into the model path
        with self._huggingface_hub_bucketfs_model_transfer.create(
                bucketfs_location=bucketfs_location,
                model_specification=current_model_specification,
                model_path=model_path,
                token=token
        ) as downloader:
            for model in [model_factory, self._tokenizer_factory]:
                downloader.download_from_huggingface_hub(model)
            # upload model files to BucketFS
            model_tar_file_path = downloader.upload_to_bucketfs()

        return str(model_path), str(model_tar_file_path)
