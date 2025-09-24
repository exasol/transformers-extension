CREATE OR REPLACE {{ language_alias }} SCALAR SCRIPT "TE_LIST_MODELS_UDF"(
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000)
) EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_name VARCHAR(2000000),
    model_path VARCHAR(2000000),
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/