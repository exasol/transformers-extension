from exasol_transformers_extension.deployment.default_udf_parameters import (
    DEFAULT_MODEL_SPECS,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecification,
    BucketFSModelSpecificationFactory,
)
from exasol_transformers_extension.utils.in_udf_model_downloader import (
    InUDFModelDownloader,
)


# todo add docu
class InstallDefaultModelsUDF:
    """
    UDF which downloads the default models specified in default_model_specs
    from Huggingface using Huggingface's
    transformers API, and uploads them to the BucketFS, from where they can then be
    loaded without accessing Huggingface again.
    Must be called with the following Input Parameter:

    returns <sub_dir/model_name> , <path of model BucketFS>
    """

    def __init__(
        self,
        exa,
        default_model_specs=DEFAULT_MODEL_SPECS,
    ):
        self._exa = exa
        self.default_model_specs = default_model_specs

    def run(self, ctx) -> None:
        for udf_name in self.default_model_specs:
            model_path = self._download_model(self.default_model_specs[udf_name])
            ctx.emit(*model_path)

    def _download_model(
        self, model_specs: BucketFSModelSpecification
    ) -> tuple[str, str]:
        # parameters
        model_downloader = InUDFModelDownloader()
        model_path, model_tar_file_path = model_downloader.download_model(
            None, model_specs, self._exa
        )

        return model_path, model_tar_file_path
