
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    BucketFSModelSpecificationFactory, BucketFSModelSpecification,
)
from exasol_transformers_extension.utils.in_udf_model_downloader import InUDFModelDownloader

from exasol_transformers_extension.deployment.default_udf_parameters import DEFAULT_MODEL_SPECS

#todo add docu?
#todo write test? or only test InUDFModelDownloader?
#todo what about unit test for this, integration only for InUDFModelDownloader?
class InstallDefaultModelsUDF:
        """
        UDF which downloads the default models specified in #todo
        from Huggingface using Huggingface's
        transformers API, and uploads them to the BucketFS, from where they can then be
        loaded without accessing Huggingface again.
        Must be called with the following Input Parameter:

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
            default_model_specs = DEFAULT_MODEL_SPECS

            for model_specs in default_model_specs:
                model_path = self._download_model(model_specs)
                ctx.emit(*model_path)

        def _download_model(self, model_specs: BucketFSModelSpecification) -> tuple[str, str]:
            # parameters
            model_downloader = InUDFModelDownloader()
            model_path, model_tar_file_path = model_downloader.download_model(
                None,
                model_specs,
                self._exa)

            return model_path, model_tar_file_path
