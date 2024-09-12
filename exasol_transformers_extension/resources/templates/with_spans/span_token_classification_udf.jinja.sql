CREATE OR REPLACE {{ language_alias }} SET SCRIPT "TE_TOKEN_CLASSIFICATION_UDF_WITH_SPAN"(
    device_id INTEGER,
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    docid INTEGER,
    text_data_char_begin INTEGER,
    text_data_char_end INTEGER,
    aggregation_strategy VARCHAR(2000000)
    ORDER BY {{ ordered_columns | join(" ASC,") }} ASC
)EMITS (
    bucketfs_conn VARCHAR(2000000),
    sub_dir VARCHAR(2000000),
    model_name VARCHAR(2000000),
    text_data VARCHAR(2000000),
    docid INTEGER,
    text_data_char_begin INTEGER,
    text_data_char_end INTEGER,
    aggregation_strategy VARCHAR(2000000),
    start_pos INTEGER,
    end_pos INTEGER,
    word VARCHAR(2000000),
    entity VARCHAR(2000000),
    score DOUBLE,
    token_docid INTEGER,
    token_char_begin INTEGER,
    token_char_end INTEGER,
    error_message VARCHAR(2000000) ) AS

{{ script_content }}

/