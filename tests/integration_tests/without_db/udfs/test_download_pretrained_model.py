import json
import tempfile
from pathlib import Path

from exasol_transformers_extension.utils.model_specification_string import ModelSpecificationString
from tests.utils.parameters import model_params
from tests.fixtures.model_fixture import download_model_to_standard_local_save_path


def test_download_pretrained_model():
    """
    Checks the expected behavior of the 3rd party api.
    """

    with tempfile.TemporaryDirectory() as download_tmpdir:
        download_model_to_standard_local_save_path(ModelSpecificationString(model_params.tiny_model), download_tmpdir)
        url_fields = []
        for file_ in Path(download_tmpdir).iterdir():
            if str(file_).endswith(".json"):
                with open(file_, 'r') as fp:
                    desc_file = json.load(fp)
                    url_fields.append(desc_file['url'])

        assert Path(download_tmpdir).is_dir() \
               and all(model_params.tiny_model in url_ for url_ in url_fields)

