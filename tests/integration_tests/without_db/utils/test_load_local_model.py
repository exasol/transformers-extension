from typing import Union
from unittest.mock import MagicMock, create_autospec

from pathlib import Path
from transformers import AutoModel, AutoTokenizer, pipeline
import tarfile
from exasol.python_extension_common.connections.bucketfs_location import (
    create_bucketfs_location_from_conn_object)

from exasol_transformers_extension.utils.bucketfs_model_specification import BucketFSModelSpecification
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory, make_parameters_of_model_contiguous_tensors
from exasol_transformers_extension.utils.bucketfs_operations import (
    create_save_pretrained_model_path)

from tests.utils.parameters import model_params
from tests.utils.mock_connections import create_mounted_bucketfs_connection


class TestSetup:
    def __init__(self):
        self.base_model_factory: ModelFactoryProtocol = AutoModel
        self.tokenizer_factory: ModelFactoryProtocol = AutoTokenizer

        self.token = "token"
        self.model_specification = model_params.tiny_model_specs

        self.mock_current_model_specification: Union[BucketFSModelSpecification, MagicMock] = create_autospec(
            BucketFSModelSpecification)
        test_pipeline = pipeline
        self.loader = LoadLocalModel(
            test_pipeline,
            task_type="token-classification",
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


def test_load_local_model(tmp_path):
    test_setup = TestSetup()

    model_specification = test_setup.model_specification
    model_save_path = create_save_pretrained_model_path(tmp_path, model_specification)
    # download a model
    model = AutoModel.from_pretrained(model_specification.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_specification.model_name)
    make_parameters_of_model_contiguous_tensors(model)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    test_setup.loader.set_current_model_specification(current_model_specification=
                                                      test_setup.mock_current_model_specification)
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

    sub_dir_path = tmp_path / sub_dir
    with tarfile.open(str(sub_dir_path / downloaded_model_path)) as tar:
        tar.extractall(path=str(sub_dir_path))

    test_setup.loader.set_current_model_specification(current_model_specification=
                                                      test_setup.mock_current_model_specification)

    test_setup.loader._bucketfs_model_cache_dir = sub_dir_path
    test_setup.loader.load_models()
