from pathlib import Path, PurePosixPath
from typing import Union
from unittest.mock import MagicMock, create_autospec

from transformers import AutoModel, AutoTokenizer, pipeline
import tarfile

from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_transformers_extension.utils.bucketfs_operations import create_save_pretrained_model_path
from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString

from tests.utils.parameters import model_params

import tempfile

#todo rename all modelspecification strings
class TestSetup:
    def __init__(self):

        self.base_model_factory: ModelFactoryProtocol = AutoModel
        self.tokenizer_factory: ModelFactoryProtocol = AutoTokenizer

        self.token = "token"
        self.model_specification = model_params.tiny_model_specs

        self.mock_current_model_specification: Union[CurrentModelSpecification, MagicMock] = create_autospec(CurrentModelSpecification)
        test_pipeline = pipeline
        self.loader = LoadLocalModel(
                                    test_pipeline,
                                    task_name="token-classification",
                                    device="cpu",
                                    base_model_factory=self.base_model_factory,
                                    tokenizer_factory=self.tokenizer_factory
                                    )


def download_model_with_huggingface_transfer(test_setup, mock_bucketfs_location):
    model_transfer_factory = HuggingFaceHubBucketFSModelTransferSPFactory()
    downloader = model_transfer_factory.create(bucketfs_location=mock_bucketfs_location,
                                               model_specification_string=test_setup.model_specification,
                                               model_path=Path("cached_files"),
                                               token="")
    downloader.download_from_huggingface_hub(test_setup.base_model_factory)
    downloader.download_from_huggingface_hub(test_setup.tokenizer_factory)
    bucketfs_model_path = downloader.upload_to_bucketfs()

    with tarfile.open(mock_bucketfs_location.base_path / bucketfs_model_path) as tar:
        tar.extractall(path=mock_bucketfs_location.base_path / bucketfs_model_path.parent)
    return mock_bucketfs_location.base_path / bucketfs_model_path.parent


def test_load_local_model():
    test_setup = TestSetup()

    with tempfile.TemporaryDirectory() as dir:
        dir_p = Path(dir)
        model_specification_string = test_setup.model_specification
        model_save_path = create_save_pretrained_model_path(dir_p, model_specification_string)
        # download a model
        model = AutoModel.from_pretrained(model_specification_string.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_specification_string.model_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        test_setup.loader._bucketfs_model_cache_dir = model_save_path
        #todo prbs need to switch this to use set_bucketfs_model_cache_dir, or add test for set_bucketfs_model_cache_dir
        test_setup.loader.load_models(current_model_specification=test_setup.mock_current_model_specification)


def test_load_local_model_with_huggingface_model_transfer():
    test_setup = TestSetup()

    with tempfile.TemporaryDirectory() as dire:
        dir_p = Path(dire)

        mock_bucketfs_location = LocalFSMockBucketFSLocation(
            PurePosixPath(dir_p / "bucket"))

        # download a model
        downloaded_model_path = download_model_with_huggingface_transfer(
            test_setup, mock_bucketfs_location)

        test_setup.loader._bucketfs_model_cache_dir = downloaded_model_path
        test_setup.loader.load_models(current_model_specification=test_setup.mock_current_model_specification)
