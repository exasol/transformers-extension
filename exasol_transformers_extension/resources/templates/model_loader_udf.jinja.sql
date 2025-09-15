CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_MODEL_LOADER_UDF"(
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    bfs_conn VARCHAR(2000000)
) EMITS (
    task VARCHAR(2000000),
    framework VARCHAR(2000000),
    device VARCHAR(2000000)
) AS

{{ script_content }}

/
