from pathlib import Path
from test.utils.parameters import model_params
from test.utils.matchers import suffix

from exasol_transformers_extension.upload_model import upload_model_to_bfs_location


def test_model_upload_python_api(tmpdir_factory):
    # get specs for a valid huggingface model
    model_specs = model_params.token_model_specs
    sub_dir = Path("subdir")

    mock_bucketfs_location = tmpdir_factory.mktemp("test_upload_python_api")
    # real bucketfs would create these dirs, but tempdir does not
    mock_bucketfs_location.mkdir(sub_dir)
    mock_bucketfs_location.mkdir(sub_dir / "dslim")

    actual_tar_path = upload_model_to_bfs_location(
        model_name=model_specs.model_name,
        task_type=model_specs.task_type,
        subdir=sub_dir,
        bucketfs_location=mock_bucketfs_location,
    )

    expected_tar_path = sub_dir / (
        model_specs.get_model_specific_path_suffix()
    ).with_suffix(".tar.gz")

    assert (mock_bucketfs_location / expected_tar_path).exists()
    assert suffix(actual_tar_path) == expected_tar_path
