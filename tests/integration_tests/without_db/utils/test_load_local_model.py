from pathlib import Path
from transformers import AutoModel, AutoTokenizer

from exasol_transformers_extension.utils.load_local_model import LoadLocalModel

from tests.utils.parameters import model_params

from tests.fixtures.bucketfs_fixture import bucketfs_location

import tempfile


class TestSetup:
    def __init__(self, bucketfs_location):
        self.bucketfs_location = bucketfs_location # do with this?

        self.token = "token"
        model_params_ = model_params.tiny_model
        self.model_name = model_params_

        self.mock_current_model_key = None
        mock_pipeline = lambda task_name, model, tokenizer, device, framework: None
        self.loader = LoadLocalModel(
            mock_pipeline,
            task_name="test_task",
            device=0)


def test_integration(bucketfs_location):
    test_setup = TestSetup(bucketfs_location)

    with tempfile.TemporaryDirectory() as dir:
        dir_p = Path(dir)
        model_save_path = dir_p / "pretrained" / test_setup.model_name
        # download a model
        model = AutoModel.from_pretrained(test_setup.model_name)
        tokenizer = AutoTokenizer.from_pretrained(test_setup.model_name)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        test_setup.loader.load_models(model_name=test_setup.model_name,
                                      current_model_key=test_setup.mock_current_model_key,
                                      cache_dir=dir_p)
