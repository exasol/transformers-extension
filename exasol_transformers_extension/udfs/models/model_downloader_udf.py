
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecificationFactory,
)
from exasol_transformers_extension.utils.in_udf_model_downloader import InUDFModelDownloader



class ModelDownloaderUDF:
    """
    UDF which downloads a pretrained model from Huggingface using Huggingface's
    transformers API, and uploads it to the BucketFS, from where it can then be
    loaded without accessing Huggingface again.
    Must be called with the following Input Parameter:


    bucketfs_conn            | sub_dir                 | model_name                 | token_conn
    ----------------------------------------------------------------------------------------------------
    BucketFS connection | directory to save model | name of Huggingface model | name of token connection

    returns <sub_dir/model_name> , <path of model BucketFS>
    """

    def __init__(
        self,
        exa,
        current_model_specification_factory: BucketFSModelSpecificationFactory = BucketFSModelSpecificationFactory(),

    ):
        self._exa = exa
        self._current_model_specification_factory = current_model_specification_factory

    def run(self, ctx) -> None:
        while True:
            model_path = self._download_model(ctx)
            ctx.emit(*model_path)
            if not ctx.next():
                break

    def _download_model(self, ctx) -> tuple[str, str]:
        # parameters
        bucketfs_conn = ctx.bucketfs_conn  # BucketFS connection
        token_conn = ctx.token_conn  # name of token connection
        current_model_specification = self._current_model_specification_factory.create(
            ctx.model_name, ctx.task_type, bucketfs_conn, ctx.sub_dir
        )  # specifies details of Huggingface model
        model_downloader = InUDFModelDownloader()
        model_path, model_tar_file_path = model_downloader.download_model(
            token_conn,
            current_model_specification,
            self._exa)

        return model_path, model_tar_file_path
