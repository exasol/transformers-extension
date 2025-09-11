"""HuggingFaceHubBucketFSModelTransferSP transfers a model from HuggingFace Hub
to the BucketFS."""

from pathlib import Path
from typing import Optional

import exasol.bucketfs as bfs
from transformers import AutoTokenizer

from exasol_transformers_extension.utils.bucketfs_model_uploader import (
    BucketFSModelUploaderFactory,
)
from exasol_transformers_extension.utils.bucketfs_operations import (
    create_save_pretrained_model_path,
)
from exasol_transformers_extension.utils.model_factory_protocol import (
    ModelFactoryProtocol,
)
from exasol_transformers_extension.utils.model_specification import ModelSpecification
from exasol_transformers_extension.utils.temporary_directory_factory import (
    TemporaryDirectoryFactory,
)


def download_transformers_model(
    bucketfs_location: bfs.path.PathLike,
    sub_dir: str,
    task_type: str,
    model_name: str,
    model_factory,
    tokenizer_factory=AutoTokenizer,
    huggingface_token: str | None = None,
) -> bfs.path.PathLike:
    """
    Downloads the specified model from the Huggingface hub into the BucketFS.
    Returns BucketFS location where the model is uploaded.

    Note: This function should NOT be called from a UDF.

    Parameters:
        bucketfs_location:
            Root location in the BucketFS.
        sub_dir:
            Root subdirectory in the BucketFS location where all models are uploaded.
        task_type:
            Name of an NLP task recognized by the Huggingface pipeline(). See
            https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/pipelines#transformers.pipeline.task
        model_name:
            Name of the model. This is the same name as it's seen on the Haggingface
            model card, for example 'cross-encoder/nli-deberta-base'.
        model_factory:
            The model class (AutoModelXXX), e.g. AutoModelForTokenClassification.
        tokenizer_factory:
            The tokenizer class, e.g. AutoTokenizer
        huggingface_token:
            Optional Huggingface token, required for downloading a private mode.
    """
    model_spec = BucketFSModelSpecification(model_name, task_type, "", Path(sub_dir))

    # Get model path in the BucketFS
    model_path = model_spec.get_bucketfs_model_save_path()

    # Download the model and the tokenizer into the model path
    with HuggingFaceHubBucketFSModelTransferSP(
        bucketfs_location=bucketfs_location,
        model_specification=model_spec,
        bucketfs_model_path=model_path,
        token=huggingface_token,
    ) as downloader:
        for factory in [model_factory, tokenizer_factory]:
            downloader.download_from_huggingface_hub(factory)
        upload_path = downloader.upload_to_bucketfs()
    return bucketfs_location / upload_path


def make_parameters_of_model_contiguous_tensors(model):
    """Fix for "ValueError: You are trying to save a non-contiguous tensor"
    when calling save_pretrained."""
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.data = param.data.contiguous()


class HuggingFaceHubBucketFSModelTransferSP:
    """
    Class for downloading a model using the Huggingface Transformers API,
    saving it locally using transformers save_pretrained,
    and loading the saved model files into the BucketFS.

    :bucketfs_location:                 BucketFSLocation the model should be loaded to
    :model_specification:               Holds information specifying details of
                                        Huggingface model to be downloaded
    :bucketfs_model_path:               Path the model will be loaded into the BucketFS
    :token:                             at Huggingface token, only needed for private
                                        models
    :temporary_directory_factory:       Optional. Default is TemporaryDirectoryFactory.
                                        Mainly change for testing.
    :bucketfs_model_uploader_factory:   Optional.
                                        Default is BucketFSModelUploaderFactory.
                                        Mainly change for testing.
    """

    def __init__(
        self,
        bucketfs_location: bfs.path.PathLike,
        model_specification: ModelSpecification,
        bucketfs_model_path: Path,
        token: Optional[str],
        temporary_directory_factory: TemporaryDirectoryFactory = TemporaryDirectoryFactory(),
        bucketfs_model_uploader_factory: BucketFSModelUploaderFactory = BucketFSModelUploaderFactory(),
    ):
        self._token = token
        self._model_specification = model_specification
        self._temporary_directory_factory = temporary_directory_factory
        self._bucketfs_model_uploader = bucketfs_model_uploader_factory.create(
            model_path=bucketfs_model_path, bucketfs_location=bucketfs_location
        )
        self._tmpdir = temporary_directory_factory.create()
        self._tmpdir_name = Path(self._tmpdir.__enter__())
        self._save_pretrained_model_path = create_save_pretrained_model_path(
            self._tmpdir_name, self._model_specification
        )

    def __enter__(self):
        return self

    def __del__(self):
        self._tmpdir.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tmpdir.__exit__(exc_type, exc_val, exc_tb)

    def download_from_huggingface_hub(self, model_factory: ModelFactoryProtocol):
        """
        Download a model from HuggingFace Hub into a temporary directory and save
        it with save_pretrained in a temporary directory.
        """
        model_name = self._model_specification.model_name
        model = model_factory.from_pretrained(
            model_name, cache_dir=self._tmpdir_name / "cache", token=self._token
        )
        make_parameters_of_model_contiguous_tensors(model)
        model.save_pretrained(self._save_pretrained_model_path)

    def upload_to_bucketfs(self) -> Path:
        """
        Upload the downloaded models into the BucketFS.

        returns: Path of the uploaded model in the BucketFS
        """
        return self._bucketfs_model_uploader.upload_directory(
            self._save_pretrained_model_path
        )


class HuggingFaceHubBucketFSModelTransferSPFactory:
    """
    Class for creating a HuggingFaceHubBucketFSModelTransferSP object.
    """

    def create(
        self,
        bucketfs_location: bfs.path.PathLike,
        model_specification: ModelSpecification,
        model_path: Path,
        token: Optional[str],
    ) -> HuggingFaceHubBucketFSModelTransferSP:
        """
        Creates a HuggingFaceHubBucketFSModelTransferSP object.

        :bucketfs_location:     BucketFSLocation the model should be loaded to
        :model_specification:   Holds information specifying details of
                                Huggingface model
        :model_path:            Path the model will be loaded into the BucketFS at
        :token:                 Huggingface token, only needed for private models

        returns: The created HuggingFaceHubBucketFSModelTransferSP object.
        """
        return HuggingFaceHubBucketFSModelTransferSP(
            bucketfs_location=bucketfs_location,
            model_specification=model_specification,
            bucketfs_model_path=model_path,
            token=token,
        )
