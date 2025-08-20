CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_TOKEN_CLASSIFICATION_UDF"(
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
) EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    task_name VARCHAR(2000000),
    path VARCHAR(2000000),
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/