import tempfile
from pathlib import Path
from typing import Union
from unittest.mock import create_autospec, MagicMock, call

from exasol_bucketfs_utils_python.bucketfs_location import BucketFSLocation

from exasol_transformers_extension.utils.model_factory_protocol import ModelFactoryProtocol
from exasol_transformers_extension.utils.load_local_model import LoadLocalModel

from tests.utils.parameters import model_params


class TestSetup:
    def __init__(self):

        self.bucketfs_location_mock: Union[BucketFSLocation, MagicMock] = create_autospec(BucketFSLocation)

        self.token = "token"
        model_params_ = model_params.tiny_model
        self.model_name = model_params_

        mock_pipeline = lambda task_name, model, tokenizer, device, framework: None #todo do we want a pipeline and check creation?
        self.loader = LoadLocalModel(
                 mock_pipeline,
                 task_name="test_task",
                 device=0)


#todo test current model key? test load model twice, test wrong model given
def test_load_function_call():
    test_setup = TestSetup()
    mock_current_model_key = None
    with tempfile.TemporaryDirectory() as dir:
        dir_p = Path(dir)
        cache_dir = dir_p
        model_save_path = Path(cache_dir) / "pretrained" / test_setup.model_name

        test_setup.loader.load_models(model_name=test_setup.model_name,
                                      current_model_key=mock_current_model_key,
                                      cache_dir=cache_dir)

        #assert test_setup.model_factory_mock.mock_calls == [
        #    call.from_pretrained(str(model_save_path))]

