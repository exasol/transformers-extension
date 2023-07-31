from pathlib import Path
from exasol_transformers_extension.utils import bucketfs_operations
from tests.utils.parameters import model_params

SUB_DIR = "test_downloader_udf_sub_dir{id}"


def test_model_downloader_udf_script_private_model(
        setup_database, pyexasol_connection, bucketfs_location):
    bucketfs_conn_name, schema_name = setup_database
    n_rows = 2
    model_paths = []
    input_data = []
    for i in range(n_rows):
        sub_dir = SUB_DIR.format(id=i)
        model_paths.append(bucketfs_operations.get_model_path(
            sub_dir, model_params.tiny_model))
        input_data.append((
            model_params.tiny_model,
            sub_dir,
            bucketfs_conn_name,
            ''
        ))

    bucketfs_files = []
    try:
        query = (
            f"SELECT TE_MODEL_DOWNLOADER_UDF("
            f"t.model_name, "
            f"t.sub_dir, "
            f"t.bucketfs_conn_name, "
            f"t.token_conn_name"
            f") FROM (VALUES {str(tuple(input_data))} AS "
            f"t(model_name, sub_dir, bucketfs_conn_name, token_conn_name));"
        )

        # execute downloader UDF
        result = pyexasol_connection.execute(query).fetchall()

        # assertions
        for i in range(n_rows):
            bucketfs_files.append(
                bucketfs_location.list_files_in_bucketfs(str(model_paths[i])))
        assert all(i[0] == str(j) for i, j in zip(result, model_paths)) and \
               all(bucketfs_files)
    finally:
        # revert, delete downloaded model files
        for i, file_ in enumerate(bucketfs_files):
            try:
                bucketfs_location.delete_file_in_bucketfs(
                    str(Path(model_paths[i], file_)))
            except Exception as exc:
                print(f"Error while deleting downloaded files, {str(exc)}")
