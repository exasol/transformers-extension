CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_DELETE_MODEL_UDF"(
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    bfs_conn VARCHAR(2000000)
) EMITS (
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    bfs_conn VARCHAR(2000000),
    success BOOLEAN,
    err_msg VARCHAR(2000000)
) AS

{{ script_content }}

/