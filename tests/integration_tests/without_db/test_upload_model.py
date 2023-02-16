import pytest
import transformers
from click.testing import CliRunner
from exasol_transformers_extension import upload_model
from exasol_transformers_extension.utils import bucketfs_operations
from tests.utils import postprocessing
from tests.utils.parameters import bucketfs_params, model_params


@pytest.fixture(scope='function')
def download_sample_models(tmp_path):
    for downloader in [transformers.AutoModel, transformers.AutoTokenizer]:
        downloader.from_pretrained(model_params.base_model, cache_dir=tmp_path)

    yield tmp_path


def test_model_upload(download_sample_models, bucketfs_location):
    sub_dir = 'sub_dir'
    download_path = download_sample_models
    upload_path = str(bucketfs_operations.get_model_path(
        sub_dir, model_params.base_model))

    args_list = [
        "--bucketfs-name", bucketfs_params.name,
        "--bucketfs-host", bucketfs_params.host,
        "--bucketfs-port", bucketfs_params.port,
        "--bucketfs_use-https", False,
        "--bucketfs-user", bucketfs_params.user,
        "--bucketfs-password", bucketfs_params.password,
        "--bucket", bucketfs_params.bucket,
        "--path-in-bucket", bucketfs_params.path_in_bucket,
        "--model-name", model_params.base_model,
        "--sub-dir", sub_dir,
        "--model-path", str(download_path),
        "--tokenizer-path", str(download_path)
    ]

    try:
        runner = CliRunner()
        result = runner.invoke(upload_model.main, args_list)
        assert result.exit_code == 0

        downloaded_files = set(
            i.name for i in download_path.iterdir() if i.is_file())
        uploaded_files = set(
            i for i in bucketfs_location.list_files_in_bucketfs(upload_path))
        assert uploaded_files == downloaded_files
    finally:
        postprocessing.cleanup_buckets(bucketfs_location, upload_path)
