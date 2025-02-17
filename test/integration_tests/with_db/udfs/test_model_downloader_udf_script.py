from pathlib import Path


from exasol_transformers_extension.utils.bucketfs_model_specification import get_BucketFSModelSpecification_from_model_Specs
from test.utils import postprocessing
from test.utils.parameters import model_params
from test.utils.bucketfs_file_list import get_bucketfs_file_list

SUB_DIR = "test_downloader_udf_sub_dir{id}"


def test_model_downloader_udf_script(
        setup_database, db_conn, bucketfs_location):
    bucketfs_conn_name, _ = setup_database
    n_rows = 2
    sub_dirs = []
    model_paths = []
    input_data = []
    for i in range(n_rows):
        sub_dir = SUB_DIR.format(id=i)
        sub_dirs.append(sub_dir)
        current_model_specs = get_BucketFSModelSpecification_from_model_Specs(model_params.tiny_model_specs,
                                                                              bucketfs_conn_name, Path(sub_dir))
        model_paths.append(current_model_specs.get_bucketfs_model_save_path())
        input_data.append((
            current_model_specs.model_name,
            current_model_specs.task_type,
            sub_dir,
            bucketfs_conn_name,
            ''
        ))

    bucketfs_files = []
    try:
        query = f"""
            SELECT TE_MODEL_DOWNLOADER_UDF(
            t.model_name,
            t.task_type,
            t.sub_dir,
            t.bucketfs_conn_name,
            t.token_conn_name
            ) FROM (VALUES {str(tuple(input_data))} AS
            t(model_name, task_type, sub_dir, bucketfs_conn_name, token_conn_name));
            """

        # execute downloader UDF
        result = db_conn.execute(query).fetchall()

        # assertions
        for i in range(n_rows):
            sub_dir_location = bucketfs_location / sub_dirs[i]
            bucketfs_files.append(get_bucketfs_file_list(sub_dir_location))

        expected_result = [(str(model_path), str(model_path.with_suffix(".tar.gz")))
                           for index, model_path in enumerate(model_paths)]
        expected_bfs_files = [[str(model_path.relative_to(sub_dirs[index]).with_suffix(".tar.gz"))]
                              for index, model_path in enumerate(model_paths)]

        assert result == expected_result
        assert bucketfs_files == expected_bfs_files
    finally:
        for sub_dir in sub_dirs:
            postprocessing.cleanup_buckets(bucketfs_location, sub_dir)
