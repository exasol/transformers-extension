CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_SEQUENCE_CLASSIFICATION_TEXT_PAIR_UDF"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    first_text VARCHAR(2000000),
    second_text VARCHAR(2000000)
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    first_text VARCHAR(2000000),
    second_text VARCHAR(2000000),
    label VARCHAR(2000000),
    score DOUBLE,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/