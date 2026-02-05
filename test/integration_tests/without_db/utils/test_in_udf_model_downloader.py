
from pathlib import (
    Path,
    PosixPath,
)

from exasol_transformers_extension.utils.in_udf_model_downloader import InUDFModelDownloader
from test.utils.mock_connections import create_mounted_bucketfs_connection
from test.utils.parameters import model_params

from test.integration_tests.without_db.udfs.utils.mock_exa_environment import (
    MockExaEnvironment,
)
from exasol_transformers_extension.utils.bucketfs_model_specification import (
    get_BucketFSModelSpecification_from_model_Specs,
)


def test_in_udf_model_downloader_test(
        tmpdir_factory
):
    bucketfs_conn_name = "bucketfs_connection"
    sub_dir = "sub_dir"
    tiny_model_specs = model_params.tiny_model_specs
    bfs_model_specs = get_BucketFSModelSpecification_from_model_Specs(
        tiny_model_specs, bucketfs_conn_name, Path(sub_dir)
    )
    expected_upload_path = bfs_model_specs.get_bucketfs_model_save_path()


    bucketfs_base_path = tmpdir_factory.mktemp("bfs_base_path")
    bucketfs_connection = create_mounted_bucketfs_connection(bucketfs_base_path)

    mock_exa = MockExaEnvironment({bucketfs_conn_name: bucketfs_connection})


    model_downloader = InUDFModelDownloader()
    model_path, model_tar_file_path = model_downloader.download_model(token_conn=None,
                                                                      model_specs=bfs_model_specs,
                                                                      exa = mock_exa)


    expected_bucketfs_upload_location = bucketfs_base_path / expected_upload_path.with_suffix(
            ".tar.gz"
        )
    assert expected_bucketfs_upload_location.isfile()
    assert expected_upload_path == PosixPath(model_path)
    assert str(model_tar_file_path) in str(expected_bucketfs_upload_location)

