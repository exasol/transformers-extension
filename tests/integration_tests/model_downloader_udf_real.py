import pathlib
from tests.utils.parameters import bucketfs_params
from exasol_bucketfs_utils_python import list_files, delete


def test_mode_downloader_udf_real(
        upload_language_container, setup_database,
        pyexasol_connection, bucket_config):

    bucketfs_conn_name, schema_name = setup_database
    model_name = 'bert-base-uncased'
    model_path = model_name.replace("-", "_")

    try:
        # execute downloader UDF
        result = pyexasol_connection.execute(
            f"SELECT TE_MODEL_DOWNLOADER_UDF("
            f"'{model_name}', '{bucketfs_conn_name}');").fetchall()

        # assertions
        path_in_the_bucket = pathlib.PurePath("container", f"{model_path}")
        files = list_files.list_files_in_bucketfs(
            bucket_config, path_in_the_bucket)
        assert result[0][0] == model_path and files
    finally:
        # revert, delete donwloaded model files
        try:
            for file_ in files:
                delete.delete_file_in_bucketfs(
                    bucket_config, pathlib.PurePath(path_in_the_bucket, file_))
        except Exception as exc:
            print(f"Error while deleting donwloaded files, {str(exc)}")
