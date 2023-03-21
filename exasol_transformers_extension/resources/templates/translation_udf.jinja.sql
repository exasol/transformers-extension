CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_TRANSLATION_UDF"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    source_language VARCHAR(2000000),
    target_language VARCHAR(2000000),
    max_length INTEGER
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    source_language VARCHAR(2000000),
    target_language VARCHAR(2000000),
    max_length INTEGER,
    translation_text VARCHAR(2000000),
    error_message VARCHAR(2000000)) AS

{{ script_content }}

/