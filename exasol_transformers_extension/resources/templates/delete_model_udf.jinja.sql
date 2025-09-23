CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_DELETE_MODEL_UDF"(
    bfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000)
) EMITS (
    bfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    success BOOLEAN,
    error_message VARCHAR(2000000)
) AS

{{ script_content }}

/