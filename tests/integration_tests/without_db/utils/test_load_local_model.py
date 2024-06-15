from pathlib import Path, PurePosixPath
from transformers import AutoModel, AutoTokenizer, pipeline
import tarfile

from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.huggingface_hub_bucketfs_model_transfer_sp import \
    HuggingFaceHubBucketFSModelTransferSPFactory
from exasol_bucketfs_utils_python.localfs_mock_bucketfs_location import \
    LocalFSMockBucketFSLocation
from exasol_transformers_extension.utils.bucketfs_operations import (
    create_save_pretrained_model_path, create_bucketfs_location_from_conn_object)

from tests.utils.parameters import model_params
from tests.utils.mounted_bucketfs_connection import create_mounted_bucketfs_connection

import tempfile


class TestSetup:
    def __init__(self):

        self.base_model_factory: ModelFactoryProtocol = AutoModel
        self.tokenizer_factory: ModelFactoryProtocol = AutoTokenizer

        self.token = "token"
        model_params_ = model_params.tiny_model
        self.model_name = model_params_

        self.mock_current_model_key = None
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
                                               model_name=test_setup.model_name,
                                               model_path=Path("cached_files"),
                                               token="")
    downloader.download_from_huggingface_hub(test_setup.base_model_factory)
    downloader.download_from_huggingface_hub(test_setup.tokenizer_factory)
    return downloader.upload_to_bucketfs()


def test_load_local_model():
    test_setup = TestSetup()

    with tempfile.TemporaryDirectory() as dir:
        dir_p = Path(dir)
        model_save_path = create_save_pretrained_model_path(dir_p, test_setup.model_name)
        # download a model
        model = AutoModel.from_pretrained(test_setup.model_name)
        tokenizer = AutoTokenizer.from_pretrained(test_setup.model_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        test_setup.loader.load_models(current_model_key=test_setup.mock_current_model_key,
                                      model_path=model_save_path)


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

    test_setup.loader.load_models(current_model_key=test_setup.mock_current_model_key,
                                  model_path=sub_dir_path)
