import json
import tempfile
import transformers
from pathlib import Path


def test_download_pretrained_model():
    """
    Checks the expected behavior of the 3rd party api.
    """

    model_name = 'bert-base-uncased'
    with tempfile.TemporaryDirectory() as tmpdir_name:
        transformers.AutoModel.from_pretrained(
            model_name, cache_dir=tmpdir_name)

        url_fields = []
        for file_ in Path(tmpdir_name).iterdir():
            if str(file_).endswith(".json"):
                with open(file_, 'r') as fp:
                    desc_file = json.load(fp)
                    url_fields.append(desc_file['url'])

        assert Path(tmpdir_name).is_dir() \
               and all(model_name in url_ for url_ in url_fields)

