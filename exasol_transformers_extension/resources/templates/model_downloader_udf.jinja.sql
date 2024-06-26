CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_MODEL_DOWNLOADER_UDF"(
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    bfs_conn VARCHAR(2000000),
    token_conn VARCHAR(2000000)
) EMITS (
    model_path_in_udfs VARCHAR(2000000),
    model_path_of_tar_file_in_bucketfs VARCHAR(2000000)
) AS

{{ script_content }}

/