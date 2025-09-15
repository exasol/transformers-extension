CREATE OR REPLACE {{ language_alias }} SET SCRIPT "{{ schema }}"."TE_LOAD_MODEL"(
    model_name VARCHAR(2000000),
    task_type VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    bfs_conn VARCHAR(2000000)
) RETURNS BOOL AS

{{ script_content }}

/
