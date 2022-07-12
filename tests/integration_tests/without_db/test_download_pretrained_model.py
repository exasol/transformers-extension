import json
from pathlib import Path
from tests.utils.parameters import model_params


def test_download_pretrained_model(download_model):
    """
    Checks the expected behavior of the 3rd party api.
    """

    tmpdir_name = download_model
    url_fields = []
    for file_ in Path(tmpdir_name).iterdir():
        if str(file_).endswith(".json"):
            with open(file_, 'r') as fp:
                desc_file = json.load(fp)
                url_fields.append(desc_file['url'])

    assert Path(tmpdir_name).is_dir() \
           and all(model_params.name in url_ for url_ in url_fields)

