import pathlib


def test_model_downloader_udf_script(
        upload_language_container, setup_database,
        pyexasol_connection, bucketfs_location):

    bucketfs_conn_name, schema_name = setup_database
    model_name = 'bert-base-uncased'
    model_path = model_name.replace("-", "_")
    bucketfs_files = []
    try:
        # execute downloader UDF
        result = pyexasol_connection.execute(
            f"SELECT TE_MODEL_DOWNLOADER_UDF("
            f"'{model_name}', '{bucketfs_conn_name}');").fetchall()

        # assertions
        bucketfs_files = bucketfs_location.list_files_in_bucketfs(model_path)
        assert result[0][0] == model_path and bucketfs_files
    finally:
        # revert, delete downloaded model files
        try:
            for file_ in bucketfs_files:
                bucketfs_location.delete_file_in_bucketfs(
                    pathlib.PurePath(model_path, file_))
        except Exception as exc:
            print(f"Error while deleting downloaded files, {str(exc)}")
