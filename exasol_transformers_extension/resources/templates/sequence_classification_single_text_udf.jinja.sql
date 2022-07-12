CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_SEQUENCE_CLASSIFICATION_SINGLE_TEXT_UDF"(
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000)
    ORDER BY model_name
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE ) AS

{{ script_content }}

/