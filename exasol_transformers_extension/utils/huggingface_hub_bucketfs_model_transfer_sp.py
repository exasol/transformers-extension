from pathlib import Path

from exasol_bucketfs_utils_python.abstract_bucketfs_location import AbstractBucketFSLocation
from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation
from exasol_transformers_extension.utils.bucketfs_operations import create_save_pretrained_model_path
from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString

from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.bucketfs_model_uploader import BucketFSModelUploaderFactory
from exasol_transformers_extension.utils.temporary_directory_factory import TemporaryDirectoryFactory


class HuggingFaceHubBucketFSModelTransferSP:
    """
    Class for downloading a model using the Huggingface Transformers API, saving it locally using
    transformers save_pretrained, and loading the saved model files into the BucketFS.

    :bucketfs_location:                 BucketFSLocation the model should be loaded to
    :model_specification_string:        Holds information specifying details of Huggingface model to be downloaded
    :model_path:                        Path the model will be loaded into the BucketFS at
    :token:                             Huggingface token, only needed for private models
    :temporary_directory_factory:       Optional. Default is TemporaryDirectoryFactory. Mainly change for testing.
    :bucketfs_model_uploader_factory:   Optional. Default is BucketFSModelUploaderFactory. Mainly change for testing.
    """
    def __init__(self,
                 bucketfs_location: BucketFSLocation,
                 model_specification_string: ModelSpecificationString,
                 bucketfs_model_path: Path,
                 token: str,
                 temporary_directory_factory: TemporaryDirectoryFactory = TemporaryDirectoryFactory(),
                 bucketfs_model_uploader_factory: BucketFSModelUploaderFactory = BucketFSModelUploaderFactory()):
        self._token = token
        self._model_specification_string = model_specification_string
        self._temporary_directory_factory = temporary_directory_factory
        self._bucketfs_model_uploader = bucketfs_model_uploader_factory.create(
            model_path=bucketfs_model_path,
            bucketfs_location=bucketfs_location)
        self._tmpdir = temporary_directory_factory.create()
        self._tmpdir_name = Path(self._tmpdir.__enter__())
        self._save_pretrained_model_path = create_save_pretrained_model_path(self._tmpdir_name,
                                                                             self._model_specification_string)

    def __enter__(self):
        return self

    def __del__(self):
        self._tmpdir.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tmpdir.__exit__(exc_type, exc_val, exc_tb)

    def download_from_huggingface_hub(self, model_factory: ModelFactoryProtocol):
        """
        Download a model from HuggingFace Hub into a temporary directory and save it with save_pretrained
        in temporary directory / pretrained / model_name.
        """
        model_name = self._model_specification_string.deconstruct()
        model = model_factory.from_pretrained(model_name, cache_dir=self._tmpdir_name / "cache",
                                              use_auth_token=self._token)
        model.save_pretrained(self._save_pretrained_model_path)

    def upload_to_bucketfs(self) -> Path:
        """
        Upload the downloaded models into the BucketFS.

        returns: Path of the uploaded model in the BucketFS
        """
        return self._bucketfs_model_uploader.upload_directory(self._save_pretrained_model_path)


class HuggingFaceHubBucketFSModelTransferSPFactory:
    """
    Class for creating a HuggingFaceHubBucketFSModelTransferSP object.
    """
    def create(self,
               bucketfs_location: BucketFSLocation,
               model_specification_string: ModelSpecificationString,
               model_path: Path,
               token: str) -> HuggingFaceHubBucketFSModelTransferSP:
        """
        Creates a HuggingFaceHubBucketFSModelTransferSP object.

        :bucketfs_location:     BucketFSLocation the model should be loaded to
        :model_specification_string:   Holds information specifying details of Huggingface model
        :model_path:            Path the model will be loaded into the BucketFS at
        :token:                 Huggingface token, only needed for private models

        returns: The created HuggingFaceHubBucketFSModelTransferSP object.
        """
        return HuggingFaceHubBucketFSModelTransferSP(bucketfs_location=bucketfs_location,
                                                     model_specification_string=model_specification_string,
                                                     bucketfs_model_path=model_path,
                                                     token=token)
