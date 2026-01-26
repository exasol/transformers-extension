from typing import Union

import exasol.python_extension_common.connections.bucketfs_location as bfs_loc

from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification

from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import (
    HuggingFaceHubBucketFSModelTransferSPFactory,
)
from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)
import transformers

#todo name download_default_models? or ai_install_defaul_models?
#todo change tests to use this instead?
#todo move to model utils?
class InUDFModelDownloader:
    """
    Class for downloading the specified model from the Huggingface hub and uploading it
    into the BucketFS.  Returns the BucketFS location where the model is
    uploaded.

    Note: This function can be called from a UDF. If you need to call from outside a UDF, use
    utils.model_utils.install_huggingface_model instead, which takes a bucketfs_location as input

    """
    def __init__(
        self,
        tokenizer_factory: ModelFactoryProtocol = transformers.AutoTokenizer,
        huggingface_hub_bucketfs_model_transfer: HuggingFaceHubBucketFSModelTransferSPFactory = HuggingFaceHubBucketFSModelTransferSPFactory(),
    ):
        self._tokenizer_factory = tokenizer_factory
        self._huggingface_hub_bucketfs_model_transfer = (
            huggingface_hub_bucketfs_model_transfer
        )


    def download_model(self, token_conn: Union[str,None], model_specs: BucketFSModelSpecification,
                       exa) -> tuple[str, str]:

        model_factory = model_specs.get_model_factory()
        # extract token from the connection if token connection name is given.
        # note that, token is required for private models. It doesn't matter
        # whether there is a token for public model or even what the token is.
        token = False
        if token_conn:
            token_conn_obj = exa.get_connection(token_conn)
            token = token_conn_obj.password

        # set model path in buckets
        model_path = model_specs.get_bucketfs_model_save_path()

        # create bucketfs location
        bucketfs_conn_obj = exa.get_connection(model_specs.bucketfs_conn_name)
        bucketfs_location = bfs_loc.create_bucketfs_location_from_conn_object(
            bucketfs_conn_obj
        )

        # download base model and tokenizer into the model path
        with self._huggingface_hub_bucketfs_model_transfer.create(
                bucketfs_location=bucketfs_location,
                model_specification=model_specs,
                model_path=model_path,
                token=token,
        ) as downloader:
            for model in [model_factory, self._tokenizer_factory]:
                downloader.download_from_huggingface_hub(model)
            # upload model files to BucketFS
            model_tar_file_path = downloader.upload_to_bucketfs()

        return str(model_path), str(model_tar_file_path)