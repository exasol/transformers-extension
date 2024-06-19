from pathlib import Path, PurePosixPath
from typing import Union
from unittest.mock import MagicMock, create_autospec

from pathlib import Path
from transformers import AutoModel, AutoTokenizer, pipeline
import tarfile

from exasol_transformers_extension.utils.current_model_specification import CurrentModelSpecification
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory
from exasol_transformers_extension.utils.bucketfs_operations import (
    create_save_pretrained_model_path, create_bucketfs_location_from_conn_object)
from exasol_transformers_extension.utils.model_specification import ModelSpecification

from tests.utils.parameters import model_params
from tests.utils.mock_connections import create_mounted_bucketfs_connection

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
                                               model_specification=test_setup.model_specification,
                                               model_path=Path("cached_files"),
                                               token="")
    downloader.download_from_huggingface_hub(test_setup.base_model_factory)
    downloader.download_from_huggingface_hub(test_setup.tokenizer_factory)
    return downloader.upload_to_bucketfs()


def test_load_local_model():
    test_setup = TestSetup()

    with tempfile.TemporaryDirectory() as dir:
        dir_p = Path(dir)
        model_specification = test_setup.model_specification
        model_save_path = create_save_pretrained_model_path(dir_p, model_specification)
        # download a model
        model = AutoModel.from_pretrained(model_specification.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_specification.model_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)


        test_setup.loader.set_current_model_specification(current_model_specification=
                                                          test_setup.mock_current_model_specification)
        #test_setup.loader.set_bucketfs_model_cache_dir(bucketfs_location=) #todo macke a mock? or add test for set_bucketfs_model_cache_dir
        test_setup.loader._bucketfs_model_cache_dir = model_save_path
        test_setup.loader.load_models()


def test_load_local_model_with_huggingface_model_transfer(tmp_path):
    test_setup = TestSetup()

    sub_dir = "bucket"

    mock_bucketfs_location = create_bucketfs_location_from_conn_object(
        create_mounted_bucketfs_connection(tmp_path, sub_dir))

    # download a model
    downloaded_model_path = download_model_with_huggingface_transfer(
        test_setup, mock_bucketfs_location)

        test_setup.loader.set_current_model_specification(current_model_specification=
                                                          test_setup.mock_current_model_specification)
        #test_setup.loader.set_bucketfs_model_cache_dir(bucketfs_location=) #todo macke a mock? or add test for set_bucketfs_model_cache_dir
        test_setup.loader._bucketfs_model_cache_dir = downloaded_model_path
        test_setup.loader.load_models()
