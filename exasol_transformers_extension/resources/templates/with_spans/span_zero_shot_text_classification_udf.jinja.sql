CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_ZERO_SHOT_TEXT_CLASSIFICATION_UDF_WITH_SPAN"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    text_data_doc_id INTEGER,
    text_data_char_begin INTEGER,
    text_data_char_end INTEGER,
    candidate_labels VARCHAR(2000000)
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data_doc_id INTEGER,
    text_data_char_begin INTEGER,
    text_data_char_end INTEGER,
    label VARCHAR(2000000),
    score DOUBLE,
    rank INTEGER,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/