CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_TOKEN_CLASSIFICATION_UDF_WITH_SPAN"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    text_data_doc_id INTEGER,
    text_data_char_begin INTEGER,
    text_data_char_end INTEGER,
    aggregation_strategy VARCHAR(2000000)
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data_doc_id INTEGER,
    text_data_char_begin INTEGER,
    text_data_char_end INTEGER,
    aggregation_strategy VARCHAR(2000000),
    entity_covered_text VARCHAR(2000000),
    entity_type VARCHAR(2000000),
    score DOUBLE,
    entity_doc_id INTEGER,
    entity_char_begin INTEGER,
    entity_char_end INTEGER,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/