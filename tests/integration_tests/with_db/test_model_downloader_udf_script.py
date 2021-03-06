from pathlib import Path
from exasol_transformers_extension.udfs import bucketfs_operations
from tests.utils.parameters import model_params


SUB_DIR = f"test_downloader_udf_sub_dir"


def test_model_downloader_udf_script(
        upload_language_container, setup_database,
        pyexasol_connection, bucketfs_location):

    bucketfs_conn_name, schema_name = setup_database
    model_path = bucketfs_operations.get_model_path(SUB_DIR, model_params.name)
    bucketfs_files = []
    try:
        # execute downloader UDF
        result = pyexasol_connection.execute(
            f"SELECT TE_MODEL_DOWNLOADER_UDF("
            f"'{model_params.name}', '{SUB_DIR}', '{bucketfs_conn_name}');")\
            .fetchall()

        # assertions
        bucketfs_files = bucketfs_location.list_files_in_bucketfs(
            str(model_path))
        assert result[0][0] == str(model_path) and bucketfs_files
    finally:
        # revert, delete downloaded model files
        for file_ in bucketfs_files:
            try:
                bucketfs_location.delete_file_in_bucketfs(
                    str(Path(model_path, file_)))
            except Exception as exc:
                print(f"Error while deleting downloaded files, {str(exc)}")
