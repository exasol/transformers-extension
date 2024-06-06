import tempfile
from pathlib import Path
from typing import Union
from unittest.mock import create_autospec, MagicMock, call, Mock

import transformers

from exasol_transformers_extension.utils.bucketfs_operations import create_save_pretrained_model_path
from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel
from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString
from tests.unit_tests.udf_wrapper_params.zero_shot.mock_zero_shot import MockPipeline
from tests.utils.mock_cast import mock_cast


class TestSetup:
    def __init__(self):

        self.model_factory_mock: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
        self.tokenizer_factory_mock: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
        self.token = "token"
        self.model_name = "model_name"
        self.mock_current_model_key = "some_key"
        self.cache_dir = "test/Path"

        self.mock_pipeline = Mock()
        self.loader = LoadLocalModel(
                                     self.mock_pipeline,
                                     task_name="test_task",
                                     device="cpu",
                                     base_model_factory=self.model_factory_mock,
                                     tokenizer_factory=self.tokenizer_factory_mock)


def test_load_function_call():
    test_setup = TestSetup()
    model_save_path = create_save_pretrained_model_path(test_setup.cache_dir, ModelSpecificationString(test_setup.model_name))

    test_setup.loader._bucketfs_model_cache_dir = model_save_path
    test_setup.loader.load_models(current_model_key=test_setup.mock_current_model_key)

    assert test_setup.model_factory_mock.mock_calls == [
        call.from_pretrained(str(model_save_path))]
    assert test_setup.tokenizer_factory_mock.mock_calls == [
        call.from_pretrained(str(model_save_path))]
    assert test_setup.mock_pipeline.mock_calls == [
        call('test_task',
             model=mock_cast(test_setup.model_factory_mock.from_pretrained).return_value,
             tokenizer=mock_cast(test_setup.tokenizer_factory_mock.from_pretrained).return_value,
             device='cpu', framework='pt')]

