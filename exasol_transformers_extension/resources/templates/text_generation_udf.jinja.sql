CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_TEXT_GENERATION_UDF"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    max_length INTEGER,
    return_full_text BOOLEAN
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    max_length INTEGER,
    return_full_text BOOLEAN,
    generated_text VARCHAR(2000000),
    error_message VARCHAR(2000000)) AS

{{ script_content }}

/