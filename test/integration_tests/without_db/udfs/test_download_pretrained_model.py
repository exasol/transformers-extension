import json
import tempfile
from pathlib import Path

from test.fixtures.model_fixture_utils import download_model_to_standard_local_save_path
from test.utils.parameters import model_params


def test_download_pretrained_model():
    """
    Checks the expected behavior of the 3rd party api.
    """

    with tempfile.TemporaryDirectory() as download_tmpdir:
        download_model_to_standard_local_save_path(model_params.tiny_model_specs, download_tmpdir)
        url_fields = []
        for file_ in Path(download_tmpdir).iterdir():
            if str(file_).endswith(".json"):
                with open(file_, 'r') as fp:
                    desc_file = json.load(fp)
                    url_fields.append(desc_file['url'])

        assert Path(download_tmpdir).is_dir() \
               and all(model_params.tiny_model_specs.model_name in url_ for url_ in url_fields)
