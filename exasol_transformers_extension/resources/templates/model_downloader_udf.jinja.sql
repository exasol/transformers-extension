CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_MODEL_DOWNLOADER_UDF"(
    model_name VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    bfs_conn VARCHAR(2000000)
)EMITS  (outputs VARCHAR(2000000)) AS

{{ script_content }}

/