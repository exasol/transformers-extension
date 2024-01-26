import tempfile
from pathlib import Path
from typing import Union
from unittest.mock import create_autospec, MagicMock, call

from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel



class TestSetup:
    def __init__(self):

        self.model_factory_mock: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
        self.tokenizer_factory_mock: Union[ModelFactoryProtocol, MagicMock] = create_autospec(ModelFactoryProtocol)
        self.token = "token"
        self.model_name = "model_name"
        self.mock_current_model_key = "some_key"
        self.cache_dir = "test/Path"

        mock_pipeline = lambda task_name, model, tokenizer, device, framework: None
        self.loader = LoadLocalModel(
                                     mock_pipeline,
                                     task_name="test_task",
                                     device=0,
                                     base_model_factory=self.model_factory_mock,
                                     tokenizer_factory=self.tokenizer_factory_mock)


def test_load_function_call():
    test_setup = TestSetup()
    model_save_path = Path(test_setup.cache_dir) / "pretrained" / test_setup.model_name

    test_setup.loader.load_models(current_model_key=test_setup.mock_current_model_key,
                                    model_path=model_save_path)

    assert test_setup.model_factory_mock.mock_calls == [
        call.from_pretrained(str(model_save_path))]
    assert test_setup.tokenizer_factory_mock.mock_calls == [
        call.from_pretrained(str(model_save_path))]

